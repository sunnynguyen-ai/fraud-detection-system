# src/models/fraud_detector.py

from typing import Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


class LogisticRegressionDetector:
    """
    Simple fraud detector using Logistic Regression inside a sklearn Pipeline.
    Public API intentionally minimal to match existing tests:
      - train(X, y)
      - predict(X)
      - predict_proba(X)
      - save(filepath)
      - load(filepath)
    """

    def __init__(self, max_iter: int = 1000, class_weight: Optional[str] = "balanced"):
        self.model: Pipeline = Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(max_iter=max_iter, class_weight=class_weight)),
            ]
        )

    def train(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Train the model and print quick evaluation metrics on a holdout set."""
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )
        self.model.fit(X_train, y_train)

        y_pred = self.model.predict(X_test)
        y_prob = self.model.predict_proba(X_test)[:, 1]

        # --- E501 fix: wrap the long classification_report call ---
        report_text = classification_report(
            y_test,
            y_pred,
            digits=4,
            zero_division=0,
        )
        print(report_text)
        print(f"AUC: {roc_auc_score(y_test, y_prob):.4f}")

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict class labels."""
        return self.model.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict fraud probability (class 1)."""
        return self.model.predict_proba(X)[:, 1]

    def save(self, filepath: str) -> None:
        """Persist the trained pipeline to disk."""
        joblib.dump(self.model, filepath)

    def load(self, filepath: str) -> None:
        """Load a trained pipeline from disk."""
        self.model = joblib.load(filepath)


def finalize_training() -> None:
    # --- F541 fix: no f-string since there are no placeholders ---
    print("Training complete.")
