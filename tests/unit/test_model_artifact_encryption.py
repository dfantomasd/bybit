"""Tests for model artifact encryption at rest (MODEL_ENCRYPT_KEY)."""

from __future__ import annotations

import numpy as np
import pytest

from trader.ml.challenger import (
    _ARTIFACT_MAGIC,
    ChallengerModel,
    decrypt_artifact,
    encrypt_artifact,
)


def _fitted_model() -> ChallengerModel:
    rng = np.random.default_rng(3)
    x = rng.normal(size=(120, 4)).astype(np.float32)
    y = (rng.random(120) < 0.3).astype(np.int32)
    model = ChallengerModel(version="v_enc", feature_names=[f"f{i}" for i in range(4)])
    model.fit_batch(x, y)
    return model


class TestArtifactEncryption:
    def test_roundtrip_with_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("MODEL_ENCRYPT_KEY", "correct horse battery staple")
        model = _fitted_model()
        blob = model.to_bytes()
        assert blob.startswith(_ARTIFACT_MAGIC)
        restored = ChallengerModel.from_bytes(blob, version="v_enc")
        assert restored.training_samples == model.training_samples
        assert restored.predict([0.1, 0.2, 0.3, 0.4]) is not None

    def test_encrypted_blob_is_not_pickle(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("MODEL_ENCRYPT_KEY", "passphrase")
        blob = _fitted_model().to_bytes()
        # joblib/pickle payloads start with b"\\x80"; the stored blob must not.
        body = blob[len(_ARTIFACT_MAGIC) :]
        assert b"\x80" != body[:1]

    def test_legacy_plain_artifact_loads_without_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("MODEL_ENCRYPT_KEY", raising=False)
        model = _fitted_model()
        plain_blob = model.to_bytes()  # no key → plaintext
        assert not plain_blob.startswith(_ARTIFACT_MAGIC)
        monkeypatch.setenv("MODEL_ENCRYPT_KEY", "now a key exists")
        restored = ChallengerModel.from_bytes(plain_blob, version="v_enc")
        assert restored.training_samples == model.training_samples

    def test_encrypted_artifact_without_key_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("MODEL_ENCRYPT_KEY", "secret-1")
        blob = _fitted_model().to_bytes()
        monkeypatch.delenv("MODEL_ENCRYPT_KEY")
        with pytest.raises(RuntimeError, match="MODEL_ENCRYPT_KEY"):
            decrypt_artifact(blob)

    def test_wrong_key_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("MODEL_ENCRYPT_KEY", "secret-1")
        blob = _fitted_model().to_bytes()
        monkeypatch.setenv("MODEL_ENCRYPT_KEY", "secret-2")
        with pytest.raises(RuntimeError, match="decryption failed"):
            decrypt_artifact(blob)

    def test_no_key_passthrough(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("MODEL_ENCRYPT_KEY", raising=False)
        data = b"plain joblib bytes"
        assert encrypt_artifact(data) == data
        assert decrypt_artifact(data) == data

    def test_ready_fernet_key_accepted(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from cryptography.fernet import Fernet

        monkeypatch.setenv("MODEL_ENCRYPT_KEY", Fernet.generate_key().decode())
        blob = _fitted_model().to_bytes()
        assert blob.startswith(_ARTIFACT_MAGIC)
        assert ChallengerModel.from_bytes(blob, version="v_enc").training_samples == 120
