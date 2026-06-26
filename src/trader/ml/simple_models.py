"""Simple ML models using only numpy/pandas - no heavy dependencies."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from decimal import Decimal
from typing import Any, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class SimpleModelState:
    """Lightweight model state for persistence."""

    model_type: str
    feature_weights: dict[str, float]
    bias: float
    feature_names: list[str]
    trained: bool


class SimpleLinearModel:
    """Simple linear regression model using numpy only."""

    def __init__(self, n_features: int = 20):
        self.n_features = n_features
        self.weights = np.zeros(n_features, dtype=np.float32)
        self.bias = 0.0
        self.feature_means = None
        self.feature_stds = None
        self.is_trained = False

    def fit(self, X: np.ndarray, y: np.ndarray, learning_rate: float = 0.01, epochs: int = 10) -> float:
        """Train using simple gradient descent."""
        if X.shape[1] != self.n_features:
            raise ValueError(f"Expected {self.n_features} features, got {X.shape[1]}")

        # Normalize features
        self.feature_means = X.mean(axis=0)
        self.feature_stds = X.std(axis=0) + 1e-8
        X_norm = (X - self.feature_means) / self.feature_stds

        # Simple linear regression via gradient descent
        for epoch in range(epochs):
            # Forward pass
            predictions = X_norm @ self.weights + self.bias

            # Compute loss and gradients
            errors = predictions - y
            loss = (errors ** 2).mean()

            # Gradient descent
            dw = (2.0 / len(y)) * (X_norm.T @ errors)
            db = (2.0 / len(y)) * errors.mean()

            self.weights -= learning_rate * dw
            self.bias -= learning_rate * db

        self.is_trained = True
        return loss

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        if not self.is_trained:
            return np.full(len(X), 0.0, dtype=np.float32)

        X_norm = (X - self.feature_means) / self.feature_stds
        return X_norm @ self.weights + self.bias

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Compute R² score."""
        predictions = self.predict(X)
        ss_res = ((y - predictions) ** 2).sum()
        ss_tot = ((y - y.mean()) ** 2).sum()
        return 1.0 - (ss_res / (ss_tot + 1e-8))


class SimpleClassifier:
    """Simple logistic regression classifier using numpy only."""

    def __init__(self, n_features: int = 20):
        self.n_features = n_features
        self.weights = np.zeros(n_features, dtype=np.float32)
        self.bias = 0.0
        self.feature_means = None
        self.feature_stds = None
        self.is_trained = False

    def sigmoid(self, z: np.ndarray) -> np.ndarray:
        """Sigmoid function."""
        return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))

    def fit(self, X: np.ndarray, y: np.ndarray, learning_rate: float = 0.01, epochs: int = 10) -> float:
        """Train using gradient descent."""
        if X.shape[1] != self.n_features:
            raise ValueError(f"Expected {self.n_features} features, got {X.shape[1]}")

        # Normalize features
        self.feature_means = X.mean(axis=0)
        self.feature_stds = X.std(axis=0) + 1e-8
        X_norm = (X - self.feature_means) / self.feature_stds

        # Ensure y is binary (0/1)
        y_binary = (y > 0.5).astype(np.float32)

        # Logistic regression via gradient descent
        for epoch in range(epochs):
            # Forward pass
            z = X_norm @ self.weights + self.bias
            predictions = self.sigmoid(z)

            # Compute loss and gradients
            errors = predictions - y_binary
            loss = -(y_binary * np.log(predictions + 1e-8) + (1 - y_binary) * np.log(1 - predictions + 1e-8)).mean()

            # Gradient descent
            dw = (1.0 / len(y)) * (X_norm.T @ errors)
            db = (1.0 / len(y)) * errors.mean()

            self.weights -= learning_rate * dw
            self.bias -= learning_rate * db

        self.is_trained = True
        return loss

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        if not self.is_trained:
            return np.full(len(X), 0.5, dtype=np.float32)

        X_norm = (X - self.feature_means) / self.feature_stds
        z = X_norm @ self.weights + self.bias
        return self.sigmoid(z)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Compute accuracy."""
        predictions = self.predict(X)
        y_binary = (y > 0.5).astype(np.float32)
        accuracy = ((predictions > 0.5) == y_binary).mean()
        return float(accuracy)


class SimpleEnsembleRegressor:
    """Ensemble of simple linear models."""

    def __init__(self, n_features: int = 20, n_models: int = 3):
        self.n_features = n_features
        self.n_models = n_models
        self.models = [SimpleLinearModel(n_features) for _ in range(n_models)]
        self.is_trained = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> float:
        """Train ensemble."""
        if X.shape[1] != self.n_features:
            raise ValueError(f"Expected {self.n_features} features, got {X.shape[1]}")

        losses = []
        for i, model in enumerate(self.models):
            # Add some noise to y to create diversity
            y_noisy = y + np.random.normal(0, 0.01 * y.std(), len(y))
            loss = model.fit(X, y_noisy, learning_rate=0.01, epochs=10)
            losses.append(loss)

        self.is_trained = True
        return float(np.mean(losses))

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict using ensemble average."""
        if not self.is_trained:
            return np.full(len(X), 0.0, dtype=np.float32)

        predictions = np.array([model.predict(X) for model in self.models])
        return predictions.mean(axis=0)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Compute R² score."""
        predictions = self.predict(X)
        ss_res = ((y - predictions) ** 2).sum()
        ss_tot = ((y - y.mean()) ** 2).sum()
        return 1.0 - (ss_res / (ss_tot + 1e-8))

    @property
    def feature_importances_(self) -> np.ndarray:
        """Get average feature importance across models."""
        importances = np.array([np.abs(model.weights) for model in self.models])
        return importances.mean(axis=0)
