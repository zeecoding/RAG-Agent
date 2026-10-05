"""Supabase JWT authentication & Neon members role-based authorization dependency."""

import base64
import json
import logging
import os
from typing import Optional

import httpx
import jwt
from fastapi import Header, HTTPException, Query, Request
from pydantic import BaseModel, Field

from app.config import settings
from app.db.pool import get_pool

logger = logging.getLogger(__name__)

ALLOWED_ROLES = frozenset(["RESPONDER", "ADMIN"])


class AuthenticatedUser(BaseModel):
    """Authenticated user context with verified organization and role."""

    id: str = Field(..., description="Internal member ID from the Neon members table.")
    supabase_user_id: str = Field(..., description="Supabase auth UUID (sub claim).")
    email: str = Field(..., description="User's email address.")
    name: str = Field(..., description="User's display name.")
    role: str = Field(..., description="Role in the organization (e.g. RESPONDER, ADMIN).")
    organization_id: str = Field(..., description="Verified organization ID.")


def _safe_decode_jwt_payload(token: str) -> dict:
    """Safely decodes JWT payload with robust base64url padding normalization.
    Works for standard JWTs as well as test tokens with missing or extra padding.
    """
    parts = token.strip().split(".")
    if len(parts) < 2:
        raise HTTPException(status_code=401, detail="Malformed JWT: expected header.payload.signature")

    payload_b64 = parts[1].rstrip("=").replace("-", "+").replace("_", "/")
    rem = len(payload_b64) % 4
    if rem > 0:
        payload_b64 += "=" * (4 - rem)

    try:
        decoded_bytes = base64.b64decode(payload_b64)
        return json.loads(decoded_bytes.decode("utf-8"))
    except Exception as e:
        raise HTTPException(
            status_code=401,
            detail=f"Unable to parse JWT payload: {str(e)}",
        ) from e


async def _verify_token_with_supabase(token: str) -> dict:
    """Verifies a JWT bearer token.

    1. Checks if it is a synthetic test token (.synthetic_sig) — only allowed when
       the TESTING environment variable is set to '1', 'true', or 'yes'.
    2. Tries local HMAC decoding if SUPABASE_JWT_SECRET is configured.
    3. Verifies against Supabase Auth /auth/v1/user endpoint.
    4. Fails closed with 503 if neither secret nor URL is configured, rather than
       silently trusting an unverified token.
    """
    token = token.strip()

    # 1. Synthetic test token bypass — ONLY permitted in test environments.
    # Gate on an explicit TESTING env var to prevent this path from being reachable
    # in production where an attacker could craft a forged base64 payload + .synthetic_sig.
    if token.endswith(".synthetic_sig"):
        _is_test_env = os.getenv("TESTING", "").lower() in ("1", "true", "yes")
        if not _is_test_env:
            logger.warning(
                "Rejected synthetic token: TESTING mode is not enabled. "
                "Synthetic tokens are only accepted in offline test environments."
            )
            raise HTTPException(
                status_code=401,
                detail="Synthetic tokens are not accepted outside of test environments.",
            )
        return _safe_decode_jwt_payload(token)

    # 2. Local HMAC verification if SUPABASE_JWT_SECRET is configured
    if settings.supabase_jwt_secret:
        try:
            return jwt.decode(
                token,
                settings.supabase_jwt_secret,
                algorithms=["HS256"],
                audience="authenticated",
            )
        except jwt.PyJWTError as e:
            logger.warning("Local JWT verification failed: %s", e)

    # 3. Remote verification against Supabase Auth endpoint
    if settings.supabase_url:
        verify_url = f"{settings.supabase_url.rstrip('/')}/auth/v1/user"
        headers = {
            "Authorization": f"Bearer {token}",
            "apikey": settings.supabase_service_role_key or "",
        }
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                res = await client.get(verify_url, headers=headers)
            if res.status_code == 200:
                user_data = res.json()
                return {
                    "sub": user_data.get("id"),
                    "email": user_data.get("email"),
                    "user_metadata": user_data.get("user_metadata", {}),
                }
            logger.warning("Supabase Auth API rejected token (%d): %s", res.status_code, res.text)
            raise HTTPException(
                status_code=401,
                detail="Invalid or expired Supabase authentication token. Please sign in again.",
            )
        except httpx.RequestError as exc:
            logger.error("Supabase Auth API unreachable: %s", exc)
            raise HTTPException(
                status_code=503,
                detail=f"Authentication service unavailable: {str(exc)}",
            )

    # 4. Fail closed: neither SUPABASE_JWT_SECRET nor SUPABASE_URL is configured.
    # Trusting an unverified token here would be a critical security hole in a
    # misconfigured production deployment. Refuse rather than silently grant access.
    logger.error(
        "Authentication misconfiguration: neither SUPABASE_JWT_SECRET nor SUPABASE_URL "
        "is set. Refusing to trust an unverified token. Configure at least one credential."
    )
    raise HTTPException(
        status_code=503,
        detail=(
            "Authentication service is not configured on this server. "
            "Contact your system administrator."
        ),
    )


async def get_current_user_with_role(
    request: Request,
    authorization: Optional[str] = Header(None, alias="Authorization"),
    x_organization_id: Optional[str] = Header(None, alias="X-Organization-Id"),
    organization_id_query: Optional[str] = Query(None, alias="organization_id"),
) -> AuthenticatedUser:
    """Validates Supabase JWT and queries Neon `members` table.

    Enforces that the user is a member of the specified organization with role
    RESPONDER or ADMIN. Any other role raises 403 Forbidden.
    """
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=401,
            detail="Missing or invalid Authorization header. Expected 'Bearer <token>'.",
        )

    raw_token = authorization.split(" ", 1)[1].strip()
    if not raw_token:
        raise HTTPException(status_code=401, detail="Empty bearer token provided.")

    payload = await _verify_token_with_supabase(raw_token)
    supabase_user_id = payload.get("sub") or payload.get("id")
    if not supabase_user_id:
        raise HTTPException(status_code=401, detail="JWT payload missing 'sub' identifier.")

    email = payload.get("email", "")

    # Target organization: Header > Query param
    target_org_id = x_organization_id or organization_id_query

    pool = get_pool()
    if pool is None:
        raise HTTPException(status_code=500, detail="Database connection pool unavailable.")

    async with pool.acquire() as conn:
        # If target_org_id was not explicitly passed, attempt to find user's primary organization
        if not target_org_id:
            row = await conn.fetchrow(
                "SELECT organization_id FROM members WHERE supabase_user_id = $1 LIMIT 1",
                supabase_user_id,
            )
            if row:
                target_org_id = row["organization_id"]
            else:
                raise HTTPException(
                    status_code=403,
                    detail="X-Organization-Id header is required and user has no organization memberships.",
                )

        member = await conn.fetchrow(
            """
            SELECT id, role, organization_id, email, name
            FROM members
            WHERE supabase_user_id = $1 AND organization_id = $2
            """,
            supabase_user_id,
            target_org_id,
        )

        if not member:
            logger.warning(
                "Unauthorized access: User %s (email: %s) is not a member of org %s",
                supabase_user_id,
                email,
                target_org_id,
            )
            raise HTTPException(
                status_code=403,
                detail=f"Forbidden: You are not a member of organization '{target_org_id}'.",
            )

        role = str(member["role"]).upper()
        if role not in ALLOWED_ROLES:
            logger.warning(
                "Forbidden role access: User %s in org %s has role '%s', required: %s",
                supabase_user_id,
                target_org_id,
                role,
                ALLOWED_ROLES,
            )
            raise HTTPException(
                status_code=403,
                detail=f"Forbidden: Insufficient permissions. Role 'RESPONDER' or 'ADMIN' required (current role: '{role}').",
            )

        return AuthenticatedUser(
            id=member["id"],
            supabase_user_id=supabase_user_id,
            email=member["email"],
            name=member["name"] or member["email"],
            role=role,
            organization_id=member["organization_id"],
        )


# Dependency alias for explicit endpoint signatures
require_responder_or_admin = get_current_user_with_role
