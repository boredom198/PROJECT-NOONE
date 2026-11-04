"""At-rest credential storage and retrieval for PROJECT NOONE.

Encrypts and decrypts exchange credentials using Windows DPAPI. Manages separate
isolated storage for Testnet and Mainnet environments under %LOCALAPPDATA%/NOONE/.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import tempfile
from typing import Optional, Tuple

from noone.security.dpapi import (
    DPAPIError,
    SecurityError,
    SecurityPlatformError,
    dpapi_protect,
    dpapi_unprotect,
)

FORMAT_VERSION: int = 1
VALID_ENVIRONMENTS: frozenset[str] = frozenset({"testnet", "mainnet"})

ENV_VAR_MAPPING: dict[str, tuple[str, str]] = {
    "testnet": ("BINANCE_TESTNET_API_KEY", "BINANCE_TESTNET_API_SECRET"),
    "mainnet": ("BINANCE_MAINNET_API_KEY", "BINANCE_MAINNET_API_SECRET"),
}


class CredentialsNotFoundError(SecurityError):
    """Raised when credentials cannot be found in storage or environment fallback."""
    pass


class CorruptedCredentialsError(SecurityError):
    """Raised when credentials payload is corrupted, tampered, or mismatched."""
    pass


def validate_environment(environment: str) -> str:
    """Validate and normalize environment identifier."""
    if not isinstance(environment, str):
        raise ValueError(f"Environment must be a string, got {type(environment).__name__}")
    env_clean = environment.strip().lower()
    if env_clean not in VALID_ENVIRONMENTS:
        raise ValueError(
            f"Invalid environment {environment!r}. Supported environments: {sorted(VALID_ENVIRONMENTS)}"
        )
    return env_clean


def get_credential_dir() -> Path:
    """Return the default NOONE credentials directory under %LOCALAPPDATA%/NOONE/."""
    local_app_data = os.environ.get("LOCALAPPDATA")
    if local_app_data:
        base = Path(local_app_data)
    else:
        base = Path.home() / "AppData" / "Local"
    return base / "NOONE"


def get_credential_file_path(environment: str, custom_dir: Optional[Path] = None) -> Path:
    """Return the filesystem path for the specified environment's encrypted credentials."""
    env_clean = validate_environment(environment)
    directory = Path(custom_dir) if custom_dir is not None else get_credential_dir()
    return directory / f"credentials_{env_clean}.enc"


def validate_credential_inputs(api_key: str, api_secret: str) -> Tuple[str, str]:
    """Validate API key and API secret inputs.

    Never includes credential values in exception messages.
    """
    if not isinstance(api_key, str) or not isinstance(api_secret, str):
        raise ValueError("API key and API secret must be strings.")
    k = api_key.strip()
    s = api_secret.strip()
    if not k:
        raise ValueError("API key cannot be empty.")
    if not s:
        raise ValueError("API secret cannot be empty.")
    return k, s


def store_credentials(
    api_key: str,
    api_secret: str,
    environment: str = "testnet",
    custom_dir: Optional[Path] = None,
) -> Path:
    """Encrypt and store credentials atomically to disk for the given environment.

    Args:
        api_key: Exchange API key string.
        api_secret: Exchange API secret string.
        environment: 'testnet' or 'mainnet'.
        custom_dir: Optional directory override (primarily for isolated unit tests).

    Returns:
        Path to the written encrypted credentials file.

    Raises:
        ValueError: If input validation fails.
        SecurityPlatformError: If not on Windows.
        DPAPIError: If DPAPI encryption fails.
        OSError: If filesystem operations fail.
    """
    env_clean = validate_environment(environment)
    clean_key, clean_secret = validate_credential_inputs(api_key, api_secret)

    payload = {
        "version": FORMAT_VERSION,
        "environment": env_clean,
        "api_key": clean_key,
        "api_secret": clean_secret,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }

    serialized_bytes = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    ciphertext = dpapi_protect(serialized_bytes)

    target_path = get_credential_file_path(env_clean, custom_dir)
    target_path.parent.mkdir(parents=True, exist_ok=True)

    # Atomic write pattern: write to temporary file in the same directory, then rename/replace
    temp_fd, temp_path_str = tempfile.mkstemp(
        prefix=f"cred_{env_clean}_",
        suffix=".tmp",
        dir=str(target_path.parent),
    )
    temp_path = Path(temp_path_str)

    try:
        with os.fdopen(temp_fd, "wb") as f:
            f.write(ciphertext)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temp_path, target_path)
    except Exception:
        if temp_path.exists():
            try:
                temp_path.unlink()
            except OSError:
                pass
        raise

    return target_path


def load_credentials(
    environment: str = "testnet",
    allow_env_fallback: bool = False,
    custom_dir: Optional[Path] = None,
) -> Tuple[str, str]:
    """Load and decrypt credentials for the specified environment.

    Args:
        environment: 'testnet' or 'mainnet'.
        allow_env_fallback: Explicit opt-in to check environment variables if file is absent.
        custom_dir: Optional directory override.

    Returns:
        Tuple of (api_key, api_secret).

    Raises:
        CredentialsNotFoundError: If file is absent (and fallback absent or disabled).
        CorruptedCredentialsError: If file is malformed, corrupted, or environment mismatches.
        SecurityPlatformError: If not on Windows.
    """
    env_clean = validate_environment(environment)
    target_path = get_credential_file_path(env_clean, custom_dir)

    if not target_path.exists():
        if allow_env_fallback:
            key_var, secret_var = ENV_VAR_MAPPING[env_clean]
            k = os.environ.get(key_var, "").strip()
            s = os.environ.get(secret_var, "").strip()
            if k and s:
                return k, s
            raise CredentialsNotFoundError(
                f"No stored credentials file found for environment {env_clean!r} at {target_path}, "
                f"and environment variables ({key_var}, {secret_var}) are missing or empty."
            )
        raise CredentialsNotFoundError(
            f"No stored credentials file found for environment {env_clean!r} at {target_path}. "
            f"Environment fallback is disabled."
        )

    try:
        with open(target_path, "rb") as f:
            ciphertext = f.read()
    except OSError as e:
        raise CredentialsNotFoundError(f"Failed to read credentials file at {target_path}: {e}") from e

    if len(ciphertext) == 0:
        raise CorruptedCredentialsError(f"Credentials file at {target_path} is empty.")

    try:
        decrypted_bytes = dpapi_unprotect(ciphertext)
    except DPAPIError as e:
        raise CorruptedCredentialsError(
            f"Failed to decrypt credentials file at {target_path}: data may be corrupted or created under another user account."
        ) from e

    try:
        payload = json.loads(decrypted_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as e:
        raise CorruptedCredentialsError(
            f"Decrypted credentials payload from {target_path} is malformed or invalid JSON."
        ) from e

    if not isinstance(payload, dict):
        raise CorruptedCredentialsError(f"Decrypted payload at {target_path} is not a valid JSON object.")

    # Validate version and environment integrity
    version = payload.get("version")
    if version != FORMAT_VERSION:
        raise CorruptedCredentialsError(
            f"Unsupported credentials format version: {version!r}. Expected {FORMAT_VERSION}."
        )

    stored_env = payload.get("environment")
    if stored_env != env_clean:
        raise CorruptedCredentialsError(
            f"Credential environment mismatch: requested {env_clean!r}, but stored payload contains {stored_env!r}."
        )

    api_key = payload.get("api_key")
    api_secret = payload.get("api_secret")

    if not isinstance(api_key, str) or not isinstance(api_secret, str) or not api_key.strip() or not api_secret.strip():
        raise CorruptedCredentialsError(
            f"Decrypted payload at {target_path} does not contain valid non-empty api_key and api_secret."
        )

    return api_key.strip(), api_secret.strip()


def has_credentials(environment: str = "testnet", custom_dir: Optional[Path] = None) -> bool:
    """Check if an encrypted credential file exists on disk for the specified environment."""
    env_clean = validate_environment(environment)
    target_path = get_credential_file_path(env_clean, custom_dir)
    return target_path.is_file() and target_path.stat().st_size > 0


def delete_credentials(environment: str = "testnet", custom_dir: Optional[Path] = None) -> bool:
    """Securely remove the encrypted credential file for the specified environment.

    Returns:
        True if the file was deleted, False if it did not exist.
    """
    env_clean = validate_environment(environment)
    target_path = get_credential_file_path(env_clean, custom_dir)
    if target_path.exists():
        try:
            target_path.unlink()
            return True
        except OSError as e:
            raise SecurityError(f"Failed to delete credentials file at {target_path}: {e}") from e
    return False

