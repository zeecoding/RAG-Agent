"""Authentication and role-based authorization package."""

from app.auth.dependencies import (
    AuthenticatedUser,
    get_current_user_with_role,
    require_responder_or_admin,
)

__all__ = [
    "AuthenticatedUser",
    "get_current_user_with_role",
    "require_responder_or_admin",
]
