"""Tests for HMAC and RSA authenticators, and webhook signature verifier."""
from __future__ import annotations

import base64
import hashlib
import hmac

import pytest

from trader.exchange.auth import HMACAuthenticator, RSAAuthenticator, verify_bybit_signature


# ---------------------------------------------------------------------------
# HMAC tests
# ---------------------------------------------------------------------------


class TestHMACAuthenticator:
    """Tests for HMACAuthenticator."""

    def _make(self, key: str = "TEST_KEY", secret: str = "TEST_SECRET") -> HMACAuthenticator:
        return HMACAuthenticator(api_key=key, api_secret=secret)

    def test_sign_produces_hex_string(self) -> None:
        auth = self._make()
        sig = auth.sign(timestamp=1700000000000, recv_window=5000, params="")
        assert isinstance(sig, str)
        assert len(sig) == 64  # SHA-256 hex is 64 chars

    def test_sign_known_test_vector(self) -> None:
        """Verify against a manually computed reference value.

        pre_sign = "{ts}{key}{rw}{params}"
        """
        api_key = "myKey"
        api_secret = "mySecret"
        timestamp = 1000000000000
        recv_window = 5000
        params = "symbol=BTCUSDT"

        pre_sign = f"{timestamp}{api_key}{recv_window}{params}"
        expected = hmac.new(
            api_secret.encode("utf-8"),
            pre_sign.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()

        auth = HMACAuthenticator(api_key=api_key, api_secret=api_secret)
        actual = auth.sign(timestamp=timestamp, recv_window=recv_window, params=params)
        assert actual == expected

    def test_sign_changes_with_different_timestamp(self) -> None:
        auth = self._make()
        sig1 = auth.sign(1000, 5000, "qty=1")
        sig2 = auth.sign(2000, 5000, "qty=1")
        assert sig1 != sig2

    def test_sign_changes_with_different_params(self) -> None:
        auth = self._make()
        sig1 = auth.sign(1000, 5000, "qty=1")
        sig2 = auth.sign(1000, 5000, "qty=2")
        assert sig1 != sig2

    def test_get_headers_contains_required_fields(self) -> None:
        auth = self._make(key="MYKEY")
        headers = auth.get_headers(timestamp=123456789, recv_window=5000, params="")
        assert headers["X-BAPI-API-KEY"] == "MYKEY"
        assert headers["X-BAPI-TIMESTAMP"] == "123456789"
        assert headers["X-BAPI-RECV-WINDOW"] == "5000"
        assert "X-BAPI-SIGN" in headers
        assert headers["Content-Type"] == "application/json"

    def test_get_headers_sign_matches_sign_method(self) -> None:
        auth = self._make()
        ts, rw, params = 999, 5000, "test=true"
        headers = auth.get_headers(timestamp=ts, recv_window=rw, params=params)
        expected_sig = auth.sign(timestamp=ts, recv_window=rw, params=params)
        assert headers["X-BAPI-SIGN"] == expected_sig

    def test_now_ms_returns_positive_int(self) -> None:
        ts = HMACAuthenticator.now_ms()
        assert isinstance(ts, int)
        assert ts > 0

    def test_no_sign_type_header_for_hmac(self) -> None:
        """HMAC auth should NOT include X-BAPI-SIGN-TYPE."""
        auth = self._make()
        headers = auth.get_headers(1, 5000, "")
        assert "X-BAPI-SIGN-TYPE" not in headers


# ---------------------------------------------------------------------------
# RSA tests
# ---------------------------------------------------------------------------

# Generate a throwaway RSA key for testing (2048-bit)
def _generate_test_rsa_pem() -> str:
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric import rsa

    private_key = rsa.generate_private_key(
        public_exponent=65537,
        key_size=2048,
    )
    return private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.TraditionalOpenSSL,
        encryption_algorithm=serialization.NoEncryption(),
    ).decode("utf-8")


_TEST_RSA_PEM = _generate_test_rsa_pem()


class TestRSAAuthenticator:
    """Tests for RSAAuthenticator."""

    def _make(self) -> RSAAuthenticator:
        return RSAAuthenticator(api_key="RSAKEY", private_key_pem=_TEST_RSA_PEM)

    def test_sign_returns_base64_string(self) -> None:
        auth = self._make()
        sig = auth.sign(timestamp=1700000000000, recv_window=5000, params="")
        assert isinstance(sig, str)
        # Should be valid base64
        decoded = base64.b64decode(sig)
        assert len(decoded) == 256  # 2048-bit RSA produces 256-byte signature

    def test_sign_differs_between_timestamps(self) -> None:
        """RSA-SHA256 is deterministic for same input but changes with different pre-sign."""
        auth = self._make()
        # Different timestamp → different pre-sign → different signature
        sig1 = auth.sign(1000, 5000, "")
        sig2 = auth.sign(2000, 5000, "")
        assert sig1 != sig2

    def test_get_headers_contains_sign_type_2(self) -> None:
        auth = self._make()
        headers = auth.get_headers(timestamp=1, recv_window=5000, params="")
        assert headers["X-BAPI-SIGN-TYPE"] == "2"

    def test_get_headers_api_key(self) -> None:
        auth = self._make()
        headers = auth.get_headers(timestamp=1, recv_window=5000, params="")
        assert headers["X-BAPI-API-KEY"] == "RSAKEY"

    def test_get_headers_has_sign(self) -> None:
        auth = self._make()
        headers = auth.get_headers(timestamp=1, recv_window=5000, params="data")
        assert "X-BAPI-SIGN" in headers
        assert len(headers["X-BAPI-SIGN"]) > 0

    def test_now_ms_returns_positive_int(self) -> None:
        ts = RSAAuthenticator.now_ms()
        assert isinstance(ts, int)
        assert ts > 0


# ---------------------------------------------------------------------------
# Webhook signature verification tests
# ---------------------------------------------------------------------------


class TestVerifyBybitSignature:
    """Tests for verify_bybit_signature."""

    def _make_sig(self, payload: str, secret: str) -> str:
        return hmac.new(
            secret.encode("utf-8"),
            payload.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()

    def test_valid_signature_returns_true(self) -> None:
        secret = "webhook_secret"
        payload = '{"event":"fill","data":{}}'
        sig = self._make_sig(payload, secret)
        assert verify_bybit_signature(payload, sig, secret) is True

    def test_invalid_signature_returns_false(self) -> None:
        secret = "webhook_secret"
        payload = '{"event":"fill","data":{}}'
        assert verify_bybit_signature(payload, "deadbeef" * 8, secret) is False

    def test_wrong_secret_returns_false(self) -> None:
        secret = "correct_secret"
        wrong_secret = "wrong_secret"
        payload = "test_payload"
        sig = self._make_sig(payload, secret)
        assert verify_bybit_signature(payload, sig, wrong_secret) is False

    def test_empty_payload_valid(self) -> None:
        secret = "s3cr3t"
        payload = ""
        sig = self._make_sig(payload, secret)
        assert verify_bybit_signature(payload, sig, secret) is True

    def test_tampered_payload_returns_false(self) -> None:
        secret = "s3cr3t"
        payload = '{"amount": 100}'
        sig = self._make_sig(payload, secret)
        tampered = '{"amount": 999}'
        assert verify_bybit_signature(tampered, sig, secret) is False
