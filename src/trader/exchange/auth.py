"""Authentication helpers for Bybit V5 API — HMAC-SHA256 and RSA-SHA256."""
from __future__ import annotations

import hashlib
import hmac
import time


# ---------------------------------------------------------------------------
# HMAC-SHA256 Authenticator
# ---------------------------------------------------------------------------


class HMACAuthenticator:
    """Signs requests using HMAC-SHA256 as per Bybit V5 specification.

    The pre-sign string format is:
        {timestamp}{api_key}{recv_window}{query_string_or_body}
    """

    def __init__(self, api_key: str, api_secret: str) -> None:
        self._api_key = api_key
        self._api_secret = api_secret.encode("utf-8")

    def sign(self, timestamp: int, recv_window: int, params: str) -> str:
        """Return hex-encoded HMAC-SHA256 signature.

        Args:
            timestamp:   Unix millisecond timestamp.
            recv_window: Request validity window in milliseconds.
            params:      Query string (GET) or JSON body string (POST).
        """
        pre_sign = f"{timestamp}{self._api_key}{recv_window}{params}"
        return hmac.new(self._api_secret, pre_sign.encode("utf-8"), hashlib.sha256).hexdigest()

    def get_headers(self, timestamp: int, recv_window: int, params: str) -> dict[str, str]:
        """Return the authentication headers required by Bybit V5.

        Args:
            timestamp:   Unix millisecond timestamp.
            recv_window: Request validity window in milliseconds.
            params:      Query string (GET) or JSON body string (POST).
        """
        signature = self.sign(timestamp, recv_window, params)
        return {
            "X-BAPI-API-KEY": self._api_key,
            "X-BAPI-TIMESTAMP": str(timestamp),
            "X-BAPI-RECV-WINDOW": str(recv_window),
            "X-BAPI-SIGN": signature,
            "Content-Type": "application/json",
        }

    @staticmethod
    def now_ms() -> int:
        """Return current Unix time in milliseconds."""
        return int(time.time() * 1000)


# ---------------------------------------------------------------------------
# RSA-SHA256 Authenticator
# ---------------------------------------------------------------------------


class RSAAuthenticator:
    """Signs requests using RSA-SHA256 (PKCS#1 v1.5).

    Bybit supports RSA authentication as an alternative to HMAC for
    API keys configured with an RSA public key.
    """

    def __init__(self, api_key: str, private_key_pem: str) -> None:
        from cryptography.hazmat.primitives import hashes, serialization
        from cryptography.hazmat.primitives.asymmetric import padding

        self._api_key = api_key
        self._padding = padding.PKCS1v15()
        self._hash_algo = hashes.SHA256()

        # Load private key once at init time
        self._private_key = serialization.load_pem_private_key(
            private_key_pem.encode("utf-8"),
            password=None,
        )

    def sign(self, timestamp: int, recv_window: int, params: str) -> str:
        """Return base64-encoded RSA-SHA256 signature.

        Args:
            timestamp:   Unix millisecond timestamp.
            recv_window: Request validity window in milliseconds.
            params:      Query string (GET) or JSON body string (POST).
        """
        import base64

        from cryptography.hazmat.primitives import hashes
        from cryptography.hazmat.primitives.asymmetric import padding

        pre_sign = f"{timestamp}{self._api_key}{recv_window}{params}"
        raw_sig = self._private_key.sign(
            pre_sign.encode("utf-8"),
            padding.PKCS1v15(),
            hashes.SHA256(),
        )
        return base64.b64encode(raw_sig).decode("utf-8")

    def get_headers(self, timestamp: int, recv_window: int, params: str) -> dict[str, str]:
        """Return authentication headers with RSA signature."""
        signature = self.sign(timestamp, recv_window, params)
        return {
            "X-BAPI-API-KEY": self._api_key,
            "X-BAPI-TIMESTAMP": str(timestamp),
            "X-BAPI-RECV-WINDOW": str(recv_window),
            "X-BAPI-SIGN": signature,
            "X-BAPI-SIGN-TYPE": "2",  # 2 = RSA
            "Content-Type": "application/json",
        }

    @staticmethod
    def now_ms() -> int:
        """Return current Unix time in milliseconds."""
        return int(time.time() * 1000)


# ---------------------------------------------------------------------------
# Webhook signature verification
# ---------------------------------------------------------------------------


def verify_bybit_signature(payload: str, signature: str, secret: str) -> bool:
    """Verify an incoming Bybit webhook signature.

    Bybit computes: HMAC-SHA256(secret, raw_body) and sends it in
    the X-Bybit-Signature header.

    Args:
        payload:   Raw request body as a string.
        signature: Hex-encoded signature from the X-Bybit-Signature header.
        secret:    Webhook secret configured in Bybit.

    Returns:
        True if the signature is valid, False otherwise.
    """
    expected = hmac.new(
        secret.encode("utf-8"),
        payload.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()
    # Use constant-time comparison to prevent timing attacks
    return hmac.compare_digest(expected, signature)
