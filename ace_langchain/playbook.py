"""
Playbook storage using LangGraph Store for persistent memory.

This implementation uses LangGraph's Store API to manage playbook bullets
as long-term memory across sessions.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Dict, List, Optional

if TYPE_CHECKING:
    from .delta import DeltaBatch

try:
    from langgraph.store import InMemoryStore, BaseStore

    LANGGRAPH_AVAILABLE = True
except ImportError:
    try:
        # Fallback for older versions
        from langgraph.store.memory import InMemoryStore
        from langgraph.store.base import BaseStore

        LANGGRAPH_AVAILABLE = True
    except ImportError:
        LANGGRAPH_AVAILABLE = False
        BaseStore = None  # type: ignore
        InMemoryStore = None  # type: ignore


@dataclass
class LangChainBullet:
    """Single playbook entry compatible with original ACE Bullet."""

    id: str
    section: str
    content: str
    helpful: int = 0
    harmful: int = 0
    neutral: int = 0
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    updated_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def apply_metadata(self, metadata: Dict[str, int]) -> None:
        """Apply metadata updates."""
        for key, value in metadata.items():
            if hasattr(self, key):
                setattr(self, key, int(value))

    def tag(self, tag: str, increment: int = 1) -> None:
        """Tag this bullet as helpful, harmful, or neutral."""
        if tag not in ("helpful", "harmful", "neutral"):
            raise ValueError(f"Unsupported tag: {tag}")
        current = getattr(self, tag)
        setattr(self, tag, current + increment)
        self.updated_at = datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> Dict:
        """Convert to dictionary for storage."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict) -> "LangChainBullet":
        """Create from dictionary."""
        return cls(**data)


class LangChainPlaybook:
    """
    Playbook implementation using LangGraph Store for persistence.

    Uses LangGraph's Store API to persist playbook bullets across sessions,
    enabling long-term memory for ACE strategies.
    """

    def __init__(self, store: Optional[BaseStore] = None, namespace: tuple = ("ace", "playbook")):
        """
        Initialize playbook with optional LangGraph store.

        Args:
            store: LangGraph store instance (defaults to InMemoryStore)
            namespace: Namespace tuple for storing bullets in the store
        """
        if not LANGGRAPH_AVAILABLE:
            raise ImportError(
                "LangGraph is required. Install with: pip install langgraph"
            )

        self.store = store or InMemoryStore()
        self.namespace = namespace
        self._bullets: Dict[str, LangChainBullet] = {}
        self._sections: Dict[str, List[str]] = {}
        self._next_id = 0

        # Load existing bullets from store
        self._load_from_store()

    def _load_from_store(self) -> None:
        """Load bullets from the store."""
        try:
            # Search for all bullets in the namespace
            items = self.store.search(self.namespace, query="")
            for item in items:
                bullet_data = item.value if hasattr(item, "value") else item
                if isinstance(bullet_data, dict):
                    bullet = LangChainBullet.from_dict(bullet_data)
                    self._bullets[bullet.id] = bullet
                    self._sections.setdefault(bullet.section, []).append(bullet.id)
        except Exception:
            # Store might be empty, that's okay
            pass

    def _save_to_store(self, bullet: LangChainBullet) -> None:
        """Save a bullet to the store."""
        self.store.put(self.namespace, bullet.id, bullet.to_dict())

    def _generate_id(self, section: str) -> str:
        """Generate a unique bullet ID."""
        self._next_id += 1
        return f"{section}_{self._next_id}"

    def add_bullet(
        self,
        section: str,
        content: str,
        bullet_id: Optional[str] = None,
        metadata: Optional[Dict[str, int]] = None,
    ) -> LangChainBullet:
        """Add a new bullet to the playbook."""
        bullet_id = bullet_id or self._generate_id(section)
        metadata = metadata or {}
        bullet = LangChainBullet(id=bullet_id, section=section, content=content)
        bullet.apply_metadata(metadata)
        self._bullets[bullet_id] = bullet
        self._sections.setdefault(section, []).append(bullet_id)
        self._save_to_store(bullet)
        return bullet

    def update_bullet(
        self,
        bullet_id: str,
        *,
        content: Optional[str] = None,
        metadata: Optional[Dict[str, int]] = None,
    ) -> Optional[LangChainBullet]:
        """Update an existing bullet."""
        bullet = self._bullets.get(bullet_id)
        if bullet is None:
            return None
        if content is not None:
            bullet.content = content
        if metadata:
            bullet.apply_metadata(metadata)
        bullet.updated_at = datetime.now(timezone.utc).isoformat()
        self._save_to_store(bullet)
        return bullet

    def tag_bullet(
        self, bullet_id: str, tag: str, increment: int = 1
    ) -> Optional[LangChainBullet]:
        """Tag a bullet as helpful, harmful, or neutral."""
        bullet = self._bullets.get(bullet_id)
        if bullet is None:
            return None
        bullet.tag(tag, increment)
        self._save_to_store(bullet)
        return bullet

    def remove_bullet(self, bullet_id: str) -> bool:
        """Remove a bullet from the playbook."""
        bullet = self._bullets.pop(bullet_id, None)
        if bullet is None:
            return False
        # Remove from section
        section_list = self._sections.get(bullet.section, [])
        if bullet_id in section_list:
            section_list.remove(bullet_id)
        # Remove from store
        try:
            self.store.delete(self.namespace, bullet_id)
        except Exception:
            pass
        return True

    def bullets(self, section: Optional[str] = None) -> List[LangChainBullet]:
        """Get all bullets, optionally filtered by section."""
        if section:
            bullet_ids = self._sections.get(section, [])
            return [self._bullets[bid] for bid in bullet_ids if bid in self._bullets]
        return list(self._bullets.values())

    def as_prompt(self) -> str:
        """Convert playbook to prompt format."""
        if not self._bullets:
            return "(empty playbook)"

        lines = []
        for section, bullet_ids in sorted(self._sections.items()):
            lines.append(f"\n=== {section} ===")
            for bullet_id in bullet_ids:
                bullet = self._bullets.get(bullet_id)
                if bullet:
                    score = bullet.helpful - bullet.harmful
                    lines.append(f"[{bullet.id}] {bullet.content} (score: {score:+d})")
        return "\n".join(lines)

    def stats(self) -> Dict[str, int]:
        """Get playbook statistics."""
        total_bullets = len(self._bullets)
        total_helpful = sum(b.helpful for b in self._bullets.values())
        total_harmful = sum(b.harmful for b in self._bullets.values())
        sections = len(self._sections)

        return {
            "total_bullets": total_bullets,
            "total_helpful": total_helpful,
            "total_harmful": total_harmful,
            "sections": sections,
        }

    def apply_delta(self, delta_batch: "DeltaBatch") -> None:
        """Apply delta operations to the playbook."""
        # DeltaBatch is imported at module level via __init__.py

        for op in delta_batch.operations:
            if op.type == "ADD":
                self.add_bullet(
                    section=op.section,
                    content=op.content or "",
                    metadata=op.metadata,
                )
            elif op.type == "UPDATE":
                if op.bullet_id:
                    self.update_bullet(
                        bullet_id=op.bullet_id,
                        content=op.content,
                        metadata=op.metadata,
                    )
            elif op.type == "TAG":
                if op.bullet_id:
                    metadata = op.metadata
                    if "helpful" in metadata:
                        self.tag_bullet(op.bullet_id, "helpful", metadata["helpful"])
                    if "harmful" in metadata:
                        self.tag_bullet(op.bullet_id, "harmful", metadata["harmful"])
                    if "neutral" in metadata:
                        self.tag_bullet(op.bullet_id, "neutral", metadata["neutral"])
            elif op.type == "REMOVE":
                if op.bullet_id:
                    self.remove_bullet(op.bullet_id)

    def save(self, path: str) -> None:
        """Save playbook to JSON file."""
        data = {
            "bullets": [b.to_dict() for b in self._bullets.values()],
            "sections": self._sections,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, path: str, store: Optional[BaseStore] = None) -> "LangChainPlaybook":
        """Load playbook from JSON file."""
        playbook = cls(store=store)
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for bullet_data in data.get("bullets", []):
            bullet = LangChainBullet.from_dict(bullet_data)
            playbook._bullets[bullet.id] = bullet
            playbook._sections.setdefault(bullet.section, []).append(bullet.id)
            playbook._save_to_store(bullet)
        return playbook

