"""PROJECT NOONE — Component 7 (Security Layer).

Provides Windows DPAPI encryption and secure at-rest credential storage.
"""
from noone.security.dpapi import (
    DPAPIError,
    NOONE_DPAPI_ENTROPY,
    SecurityError,
    SecurityPlatformError,
    dpapi_protect,
    dpapi_unprotect,
)
from noone.security.credentials import (
    CorruptedCredentialsError,
    CredentialsNotFoundError,
    delete_credentials,
    get_credential_dir,
    get_credential_file_path,
    has_credentials,
    load_credentials,
    store_credentials,
    validate_credential_inputs,
    validate_environment,
)

__all__ = [
    "CorruptedCredentialsError",
    "CredentialsNotFoundError",
    "DPAPIError",
    "NOONE_DPAPI_ENTROPY",
    "SecurityError",
    "SecurityPlatformError",
    "delete_credentials",
    "dpapi_protect",
    "dpapi_unprotect",
    "get_credential_dir",
    "get_credential_file_path",
    "has_credentials",
    "load_credentials",
    "store_credentials",
    "validate_credential_inputs",
    "validate_environment",
]

