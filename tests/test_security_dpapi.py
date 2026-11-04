"""Unit tests for Component 7 — Security Layer (Windows DPAPI & Credential Storage).

Tests low-level DPAPI encryption/decryption, at-rest credential storage, environment
isolation, integrity verification, error handling, and environment-variable fallback.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys
from unittest.mock import patch
import pytest

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

IS_WINDOWS = sys.platform == "win32"


# =============================================================================
# 1. Low-level Windows DPAPI Tests
# =============================================================================

@pytest.mark.skipif(not IS_WINDOWS, reason="Native Windows DPAPI requires Windows OS")
class TestWindowsDPAPINative:
    """Tests executing real CryptProtectData and CryptUnprotectData on Windows."""

    def test_dpapi_protect_unprotect_roundtrip(self):
        sample = b"binance_secret_key_bytes_1234567890!@#"
        ciphertext = dpapi_protect(sample)
        assert isinstance(ciphertext, bytes)
        assert ciphertext != sample
        assert len(ciphertext) > len(sample)

        decrypted = dpapi_unprotect(ciphertext)
        assert decrypted == sample

    def test_dpapi_custom_entropy_roundtrip(self):
        sample = b"test_payload_with_custom_entropy"
        custom_entropy = b"ANOTHER_SALT_987654"
        ciphertext = dpapi_protect(sample, entropy=custom_entropy)
        assert dpapi_unprotect(ciphertext, entropy=custom_entropy) == sample

    def test_dpapi_wrong_entropy_fails(self):
        sample = b"secret_data"
        ciphertext = dpapi_protect(sample, entropy=b"CORRECT_SALT")
        with pytest.raises(DPAPIError):
            dpapi_unprotect(ciphertext, entropy=b"WRONG_SALT")

    def test_dpapi_tampered_ciphertext_fails(self):
        sample = b"secret_data_to_be_tampered"
        ciphertext = bytearray(dpapi_protect(sample))
        # Tamper with internal byte
        ciphertext[len(ciphertext) // 2] ^= 0xFF
        with pytest.raises(DPAPIError):
            dpapi_unprotect(bytes(ciphertext))

    def test_dpapi_invalid_inputs(self):
        with pytest.raises(ValueError, match="must be bytes"):
            dpapi_protect("string_not_bytes")  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="cannot be empty"):
            dpapi_protect(b"")
        with pytest.raises(ValueError, match="must be bytes"):
            dpapi_unprotect("string_not_bytes")  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="cannot be empty"):
            dpapi_unprotect(b"")


class TestPlatformDetection:
    """Verify that non-Windows platforms are detected and rejected with SecurityPlatformError."""

    def test_dpapi_protect_raises_on_non_windows(self):
        with patch("sys.platform", "linux"):
            with pytest.raises(SecurityPlatformError, match="requires a Windows operating system"):
                dpapi_protect(b"some_bytes")

    def test_dpapi_unprotect_raises_on_non_windows(self):
        with patch("sys.platform", "darwin"):
            with pytest.raises(SecurityPlatformError, match="requires a Windows operating system"):
                dpapi_unprotect(b"some_bytes")


# =============================================================================
# 2. Credential Storage & Retrieval Tests
# =============================================================================

@pytest.mark.skipif(not IS_WINDOWS, reason="At-rest storage tests use native Windows DPAPI")
class TestCredentialStorageNative:
    """Verifies credential storage lifecycle, atomic writes, and environment isolation."""

    @pytest.fixture
    def temp_cred_dir(self, tmp_path: Path) -> Path:
        cred_dir = tmp_path / "NOONE_CREDS"
        cred_dir.mkdir(parents=True, exist_ok=True)
        return cred_dir

    def test_store_and_load_testnet_credentials(self, temp_cred_dir: Path):
        api_key = "testnet_key_sample_12345"
        api_secret = "testnet_secret_sample_abcdef67890"

        file_path = store_credentials(api_key, api_secret, environment="testnet", custom_dir=temp_cred_dir)
        assert file_path.is_file()
        assert file_path.name == "credentials_testnet.enc"
        assert has_credentials(environment="testnet", custom_dir=temp_cred_dir)

        loaded_key, loaded_secret = load_credentials(environment="testnet", custom_dir=temp_cred_dir)
        assert loaded_key == api_key
        assert loaded_secret == api_secret

    def test_store_and_load_mainnet_credentials(self, temp_cred_dir: Path):
        api_key = "mainnet_key_production_45678"
        api_secret = "mainnet_secret_production_fedcba09876"

        file_path = store_credentials(api_key, api_secret, environment="mainnet", custom_dir=temp_cred_dir)
        assert file_path.is_file()
        assert file_path.name == "credentials_mainnet.enc"
        assert has_credentials(environment="mainnet", custom_dir=temp_cred_dir)

        loaded_key, loaded_secret = load_credentials(environment="mainnet", custom_dir=temp_cred_dir)
        assert loaded_key == api_key
        assert loaded_secret == api_secret

    def test_separate_testnet_and_mainnet_storage(self, temp_cred_dir: Path):
        store_credentials("t_key", "t_sec", environment="testnet", custom_dir=temp_cred_dir)
        store_credentials("m_key", "m_sec", environment="mainnet", custom_dir=temp_cred_dir)

        # Confirm two distinct files exist
        assert (temp_cred_dir / "credentials_testnet.enc").exists()
        assert (temp_cred_dir / "credentials_mainnet.enc").exists()

        t_key, t_sec = load_credentials(environment="testnet", custom_dir=temp_cred_dir)
        m_key, m_sec = load_credentials(environment="mainnet", custom_dir=temp_cred_dir)

        assert t_key == "t_key"
        assert t_sec == "t_sec"
        assert m_key == "m_key"
        assert m_sec == "m_sec"

        # Deleting testnet does not affect mainnet
        assert delete_credentials(environment="testnet", custom_dir=temp_cred_dir) is True
        assert not has_credentials(environment="testnet", custom_dir=temp_cred_dir)
        assert has_credentials(environment="mainnet", custom_dir=temp_cred_dir)
        assert load_credentials(environment="mainnet", custom_dir=temp_cred_dir) == ("m_key", "m_sec")

    def test_no_plaintext_credentials_in_stored_file(self, temp_cred_dir: Path):
        api_key = "super_distinct_api_key_xyz987"
        api_secret = "ultra_confidential_secret_qwerty123"

        file_path = store_credentials(api_key, api_secret, environment="testnet", custom_dir=temp_cred_dir)
        raw_bytes = file_path.read_bytes()

        # The raw binary file on disk MUST NOT contain the plaintext key or secret
        assert api_key.encode("utf-8") not in raw_bytes
        assert api_secret.encode("utf-8") not in raw_bytes
        assert b"api_key" not in raw_bytes
        assert b"api_secret" not in raw_bytes

    def test_atomic_write_leaves_no_temp_files(self, temp_cred_dir: Path):
        store_credentials("key1", "sec1", environment="testnet", custom_dir=temp_cred_dir)
        files = list(temp_cred_dir.glob("*"))
        assert len(files) == 1
        assert files[0].name == "credentials_testnet.enc"

    def test_corrupted_ciphertext_raises_corrupted_credentials_error(self, temp_cred_dir: Path):
        file_path = temp_cred_dir / "credentials_testnet.enc"
        file_path.write_bytes(b"corrupted_garbage_not_dpapi_blob_1234567890")

        with pytest.raises(CorruptedCredentialsError, match="Failed to decrypt|data may be corrupted"):
            load_credentials(environment="testnet", custom_dir=temp_cred_dir)

    def test_empty_credentials_file_raises_corrupted_error(self, temp_cred_dir: Path):
        file_path = temp_cred_dir / "credentials_testnet.enc"
        file_path.write_bytes(b"")

        with pytest.raises(CorruptedCredentialsError, match="is empty"):
            load_credentials(environment="testnet", custom_dir=temp_cred_dir)

    def test_wrong_environment_payload_mismatch_rejected(self, temp_cred_dir: Path):
        # Store valid testnet credentials
        t_path = store_credentials("key_val", "sec_val", environment="testnet", custom_dir=temp_cred_dir)

        # Move/copy that file to credentials_mainnet.enc
        m_path = temp_cred_dir / "credentials_mainnet.enc"
        m_path.write_bytes(t_path.read_bytes())

        # Attempting to load as mainnet must reject because decrypted payload declares "environment": "testnet"
        with pytest.raises(CorruptedCredentialsError, match="Credential environment mismatch"):
            load_credentials(environment="mainnet", custom_dir=temp_cred_dir)

    def test_missing_credentials_file_raises_credentials_not_found(self, temp_cred_dir: Path):
        assert not has_credentials(environment="testnet", custom_dir=temp_cred_dir)
        with pytest.raises(CredentialsNotFoundError, match="No stored credentials file found"):
            load_credentials(environment="testnet", allow_env_fallback=False, custom_dir=temp_cred_dir)

    def test_delete_nonexistent_credentials_returns_false(self, temp_cred_dir: Path):
        assert delete_credentials(environment="testnet", custom_dir=temp_cred_dir) is False


# =============================================================================
# 3. Environment Variable Fallback Opt-in Tests
# =============================================================================

class TestEnvironmentVariableFallback:
    """Verifies that environment variable fallback only occurs when explicitly requested."""

    @pytest.fixture
    def empty_dir(self, tmp_path: Path) -> Path:
        d = tmp_path / "empty_dir"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def test_no_fallback_by_default_even_if_env_vars_present(self, empty_dir: Path, monkeypatch):
        monkeypatch.setenv("BINANCE_TESTNET_API_KEY", "env_key_test")
        monkeypatch.setenv("BINANCE_TESTNET_API_SECRET", "env_secret_test")

        # allow_env_fallback=False by default -> MUST raise CredentialsNotFoundError
        with pytest.raises(CredentialsNotFoundError, match="Environment fallback is disabled"):
            load_credentials(environment="testnet", allow_env_fallback=False, custom_dir=empty_dir)

    def test_fallback_opt_in_testnet_success(self, empty_dir: Path, monkeypatch):
        monkeypatch.setenv("BINANCE_TESTNET_API_KEY", "env_key_test")
        monkeypatch.setenv("BINANCE_TESTNET_API_SECRET", "env_secret_test")

        k, s = load_credentials(environment="testnet", allow_env_fallback=True, custom_dir=empty_dir)
        assert k == "env_key_test"
        assert s == "env_secret_test"

    def test_fallback_opt_in_mainnet_success(self, empty_dir: Path, monkeypatch):
        monkeypatch.setenv("BINANCE_MAINNET_API_KEY", "env_key_main")
        monkeypatch.setenv("BINANCE_MAINNET_API_SECRET", "env_secret_main")

        k, s = load_credentials(environment="mainnet", allow_env_fallback=True, custom_dir=empty_dir)
        assert k == "env_key_main"
        assert s == "env_secret_main"

    def test_fallback_does_not_cross_environments(self, empty_dir: Path, monkeypatch):
        monkeypatch.setenv("BINANCE_MAINNET_API_KEY", "env_key_main")
        monkeypatch.setenv("BINANCE_MAINNET_API_SECRET", "env_secret_main")
        monkeypatch.delenv("BINANCE_TESTNET_API_KEY", raising=False)
        monkeypatch.delenv("BINANCE_TESTNET_API_SECRET", raising=False)

        # Requesting testnet with allow_env_fallback=True must NOT return mainnet credentials
        with pytest.raises(CredentialsNotFoundError, match="missing or empty"):
            load_credentials(environment="testnet", allow_env_fallback=True, custom_dir=empty_dir)

    def test_fallback_missing_one_env_var_raises(self, empty_dir: Path, monkeypatch):
        monkeypatch.setenv("BINANCE_TESTNET_API_KEY", "env_key_test")
        monkeypatch.delenv("BINANCE_TESTNET_API_SECRET", raising=False)

        with pytest.raises(CredentialsNotFoundError, match="missing or empty"):
            load_credentials(environment="testnet", allow_env_fallback=True, custom_dir=empty_dir)


# =============================================================================
# 4. Input Validation & Plaintext Leakage Prevention Tests
# =============================================================================

class TestInputValidationAndLeakagePrevention:
    """Verifies that invalid inputs are rejected and credentials are never leaked in errors."""

    @pytest.mark.parametrize("invalid_env", ["", "staging", "dev", "local", "prod", "123"])
    def test_invalid_environment_rejected(self, invalid_env: str):
        with pytest.raises(ValueError, match="Invalid environment"):
            validate_environment(invalid_env)

    @pytest.mark.parametrize("bad_key", ["", "   ", None, 12345])
    def test_invalid_api_key_rejected(self, bad_key):
        with pytest.raises(ValueError):
            validate_credential_inputs(bad_key, "valid_secret_123")  # type: ignore[arg-type]

    @pytest.mark.parametrize("bad_sec", ["", "   ", None, 12345])
    def test_invalid_api_secret_rejected(self, bad_sec):
        with pytest.raises(ValueError):
            validate_credential_inputs("valid_key_123", bad_sec)  # type: ignore[arg-type]

    def test_exception_never_contains_credential_values(self):
        secret_sample = "sensitive_super_secret_payload_9999"
        try:
            validate_credential_inputs("", secret_sample)
        except ValueError as exc:
            assert secret_sample not in str(exc)


# =============================================================================
# 5. Non-Windows Mock DPAPI Tests (CI / Cross-Platform Simulation)
# =============================================================================

class TestMockedDPAPIStorageForNonWindows:
    """Verifies credential storage workflow using mocked DPAPI on non-Windows platforms.

    Note: These synthetic tests verify workflow logic only and do NOT substitute
    for real Windows DPAPI CryptProtectData verification.
    """

    @pytest.fixture
    def mock_cred_dir(self, tmp_path: Path) -> Path:
        d = tmp_path / "mock_creds"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def test_mocked_dpapi_roundtrip_flow(self, mock_cred_dir: Path):
        # A simple reversible mock transform (prefix bytes)
        def fake_protect(data: bytes, entropy: bytes = b"", description: str = "") -> bytes:
            return b"MOCK_CIPHER:" + data

        def fake_unprotect(ciphertext: bytes, entropy: bytes = b"") -> bytes:
            if not ciphertext.startswith(b"MOCK_CIPHER:"):
                raise DPAPIError("Corrupted mock ciphertext")
            return ciphertext[len(b"MOCK_CIPHER:"):]

        with patch("noone.security.credentials.dpapi_protect", side_effect=fake_protect), \
             patch("noone.security.credentials.dpapi_unprotect", side_effect=fake_unprotect):

            path = store_credentials("mock_key", "mock_sec", environment="testnet", custom_dir=mock_cred_dir)
            assert path.exists()

            k, s = load_credentials(environment="testnet", custom_dir=mock_cred_dir)
            assert k == "mock_key"
            assert s == "mock_sec"

