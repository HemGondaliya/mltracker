"""User account management."""

from __future__ import annotations

import hashlib
import uuid
from datetime import datetime, timezone
from typing import Any

from mltracker.config import Config, get_config
from mltracker.storage import StorageBackend, create_backend


class UserAccount:
    """Represents a registered user profile."""

    def __init__(self, user_id: str, username: str, email: str | None = None,
                 *, backend: StorageBackend | None = None, config: Config | None = None):
        self.user_id = user_id
        self.username = username
        self.email = email
        self._config = config or get_config()
        self._backend = backend or create_backend(self._config)

    def to_dict(self) -> dict[str, Any]:
        return {
            "user_id": self.user_id,
            "username": self.username,
            "email": self.email,
        }

    def experiments(self) -> list[dict[str, Any]]:
        """List all experiments belonging to this user."""
        return self._backend.list_experiments(user_id=self.user_id)

    def __repr__(self) -> str:
        return f"UserAccount(user_id={self.user_id!r}, username={self.username!r})"


# ── Module-level convenience functions ───────────────────────────────────────

def register(username: str, email: str | None = None, *,
             config: Config | None = None) -> UserAccount:
    """Create a new user account."""
    cfg = config or get_config()
    backend = create_backend(cfg)

    existing = backend.load_user_by_username(username)
    if existing:
        raise ValueError(f"Username '{username}' already registered.")

    user_id = uuid.uuid4().hex[:12]
    user_data = {
        "user_id": user_id,
        "username": username,
        "email": email,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    backend.save_user(user_data)
    return UserAccount(user_id, username, email, backend=backend, config=cfg)


def login(username: str, *, config: Config | None = None) -> UserAccount:
    """Retrieve an existing user or auto-register with defaults."""
    cfg = config or get_config()
    backend = create_backend(cfg)

    data = backend.load_user_by_username(username)
    if data is None:
        # Auto-create for convenience
        return register(username, config=cfg)
    return UserAccount(data["user_id"], data["username"], data.get("email"),
                       backend=backend, config=cfg)
