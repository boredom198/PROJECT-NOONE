"""Windows Data Protection API (DPAPI) wrapper for PROJECT NOONE.

Provides user-level at-rest encryption and decryption using Windows CryptProtectData
and CryptUnprotectData via ctypes without external dependencies.
"""
from __future__ import annotations

import ctypes
from ctypes import wintypes
import sys
from typing import Optional


class SecurityError(Exception):
    """Base exception for Component 7 security layer."""
    pass


class SecurityPlatformError(SecurityError):
    """Raised when operating system does not support Windows DPAPI."""
    pass


class DPAPIError(SecurityError):
    """Raised when Windows DPAPI CryptProtectData or CryptUnprotectData fails."""
    pass


# Fixed application-specific optional entropy for PROJECT NOONE.
# Prevents arbitrary third-party software running under the same user account
# from decrypting NOONE credentials without this application salt.
NOONE_DPAPI_ENTROPY: bytes = b"PROJECT_NOONE_V1_DPAPI_SALT_e9c3a1b8"


class _DATA_BLOB(ctypes.Structure):
    _fields_ = [
        ("cbData", wintypes.DWORD),
        ("pbData", ctypes.POINTER(ctypes.c_byte)),
    ]


def _check_platform() -> None:
    if sys.platform != "win32":
        raise SecurityPlatformError(
            f"Windows DPAPI requires a Windows operating system, but current platform is {sys.platform!r}."
        )


def _init_windows_crypto():
    _check_platform()
    crypt32 = ctypes.windll.crypt32
    kernel32 = ctypes.windll.kernel32

    crypt32.CryptProtectData.argtypes = [
        ctypes.POINTER(_DATA_BLOB),          # pDataIn
        wintypes.LPCWSTR,                    # szDataDescr
        ctypes.POINTER(_DATA_BLOB),          # pOptionalEntropy
        ctypes.c_void_p,                     # pvReserved
        ctypes.c_void_p,                     # pPromptStruct
        wintypes.DWORD,                      # dwFlags
        ctypes.POINTER(_DATA_BLOB),          # pDataOut
    ]
    crypt32.CryptProtectData.restype = wintypes.BOOL

    crypt32.CryptUnprotectData.argtypes = [
        ctypes.POINTER(_DATA_BLOB),          # pDataIn
        ctypes.POINTER(wintypes.LPWSTR),     # ppszDataDescr
        ctypes.POINTER(_DATA_BLOB),          # pOptionalEntropy
        ctypes.c_void_p,                     # pvReserved
        ctypes.c_void_p,                     # pPromptStruct
        wintypes.DWORD,                      # dwFlags
        ctypes.POINTER(_DATA_BLOB),          # pDataOut
    ]
    crypt32.CryptUnprotectData.restype = wintypes.BOOL

    kernel32.LocalFree.argtypes = [wintypes.HLOCAL]
    kernel32.LocalFree.restype = wintypes.HLOCAL

    return crypt32, kernel32


def dpapi_protect(
    data: bytes,
    entropy: bytes = NOONE_DPAPI_ENTROPY,
    description: str = "NOONE_CREDENTIALS",
) -> bytes:
    """Encrypt binary data using Windows CryptProtectData with current-user protection.

    Args:
        data: Plaintext bytes to encrypt.
        entropy: Application-specific optional entropy salt.
        description: Informational description string for the data blob.

    Returns:
        Encrypted binary ciphertext.

    Raises:
        SecurityPlatformError: If not running on Windows.
        DPAPIError: If CryptProtectData fails.
        ValueError: If input data is not bytes or empty.
    """
    _check_platform()
    if not isinstance(data, (bytes, bytearray)):
        raise ValueError(f"Data to protect must be bytes, got {type(data).__name__}")
    if len(data) == 0:
        raise ValueError("Data to protect cannot be empty.")

    crypt32, kernel32 = _init_windows_crypto()

    data_bytes = (ctypes.c_byte * len(data))(*data)
    data_blob = _DATA_BLOB(len(data), ctypes.cast(data_bytes, ctypes.POINTER(ctypes.c_byte)))

    entropy_ptr = None
    if entropy:
        ent_bytes = (ctypes.c_byte * len(entropy))(*entropy)
        entropy_blob = _DATA_BLOB(len(entropy), ctypes.cast(ent_bytes, ctypes.POINTER(ctypes.c_byte)))
        entropy_ptr = ctypes.byref(entropy_blob)

    out_blob = _DATA_BLOB()
    # dwFlags = 0: CRYPTPROTECT_UI_FORBIDDEN is 0x1, current-user protection is default 0
    flags = 0x1  # UI forbidden

    ret = crypt32.CryptProtectData(
        ctypes.byref(data_blob),
        description,
        entropy_ptr,
        None,
        None,
        flags,
        ctypes.byref(out_blob),
    )

    if not ret:
        err_code = ctypes.GetLastError()
        raise DPAPIError(f"Windows DPAPI encryption failed (Error Code {err_code}).")

    try:
        raw_ciphertext = ctypes.string_at(out_blob.pbData, out_blob.cbData)
        return raw_ciphertext
    finally:
        if out_blob.pbData:
            kernel32.LocalFree(out_blob.pbData)


def dpapi_unprotect(
    ciphertext: bytes,
    entropy: bytes = NOONE_DPAPI_ENTROPY,
) -> bytes:
    """Decrypt binary data using Windows CryptUnprotectData.

    Args:
        ciphertext: Encrypted binary bytes from dpapi_protect.
        entropy: Application-specific optional entropy salt matching encryption.

    Returns:
        Decrypted plaintext bytes.

    Raises:
        SecurityPlatformError: If not running on Windows.
        DPAPIError: If CryptUnprotectData fails (e.g. corrupted or wrong user).
        ValueError: If input ciphertext is not bytes or empty.
    """
    _check_platform()
    if not isinstance(ciphertext, (bytes, bytearray)):
        raise ValueError(f"Ciphertext must be bytes, got {type(ciphertext).__name__}")
    if len(ciphertext) == 0:
        raise ValueError("Ciphertext cannot be empty.")

    crypt32, kernel32 = _init_windows_crypto()

    enc_bytes = (ctypes.c_byte * len(ciphertext))(*ciphertext)
    enc_blob = _DATA_BLOB(len(ciphertext), ctypes.cast(enc_bytes, ctypes.POINTER(ctypes.c_byte)))

    entropy_ptr = None
    if entropy:
        ent_bytes = (ctypes.c_byte * len(entropy))(*entropy)
        entropy_blob = _DATA_BLOB(len(entropy), ctypes.cast(ent_bytes, ctypes.POINTER(ctypes.c_byte)))
        entropy_ptr = ctypes.byref(entropy_blob)

    out_blob = _DATA_BLOB()
    flags = 0x1  # UI forbidden

    ret = crypt32.CryptUnprotectData(
        ctypes.byref(enc_blob),
        None,
        entropy_ptr,
        None,
        None,
        flags,
        ctypes.byref(out_blob),
    )

    if not ret:
        err_code = ctypes.GetLastError()
        raise DPAPIError(
            f"Windows DPAPI decryption failed (Error Code {err_code}): data may be corrupted or created under another user account."
        )

    try:
        plaintext = ctypes.string_at(out_blob.pbData, out_blob.cbData)
        return plaintext
    finally:
        if out_blob.pbData:
            kernel32.LocalFree(out_blob.pbData)

