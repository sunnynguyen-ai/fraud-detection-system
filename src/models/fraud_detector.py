# src/models/fraud_detector.py

from typing import Dict, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


class LogisticRegressionDetector:
    """
    Simple fraud detector using Logistic Regression inside a sklearn Pipeline.

    Public API (intentionally minimal to match tests):
      - train(X, y)
      - predict(X)
      - predict_proba(X)
      - save(filepath)
      - load(filepath)
    """

    def __init__(
        self,
        max_iter: int = 1000,
        class_weight: Optional[str] = "balanced",
        balance_data: bool = False,
        random_state: int = 42,
    ):
        self.model: Pipeline = Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(max_iter=max_iter, class_weight=class_weight)),
            ]
        )
        self.balance_data = balance_data
        self.random_state = random_state

    # ---------------------------
    # Internal helpers
    # ---------------------------

    def _split_data(
        self, X: pd.DataFrame, y: pd.Series
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        return train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=self.random_state
        )

    def _maybe_balance(
        self, X_train: pd.DataFrame, y_train: pd.Series
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """Optionally apply SMOTE to balance the training set."""
        if not self.balance_data:
            return X_train, y_train

        print(f"Original training data: {y_train.value_counts().to_dict()}")

        smote = SMOTE(random_state=self.random_state)
        X_bal, y_bal = smote.fit_resample(X_train, y_train)

        balanced_counts = pd.Series(y_bal).value_counts().to_dict()
        print(f"Balanced training data: {balanced_counts}")

        return X_bal, y_bal

    # ---------------------------
    # Training / Evaluation
    # ---------------------------

    def train(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Train the model and print quick evaluation metrics on a holdout set."""
        X_train, X_test, y_train, y_test = self._split_data(X, y)
        X_train, y_train = self._maybe_balance(X_train, y_train)

        self.model.fit(X_train, y_train)

        y_pred = self.model.predict(X_test)
        y_prob = self.model.predict_proba(X_test)[:, 1]

        # Wrap long call to satisfy E501
        report_text = classification_report(
            y_test,
            y_pred,
            digits=4,
            zero_division=0,
        )
        print(report_text)
        print(f"AUC: {roc_auc_score(y_test, y_prob):.4f}")

    def evaluate_model(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
        """Return a small set of metrics; prints a classification report."""
        y_pred = self.model.predict(X_test)
        y_prob = self.model.predict_proba(X_test)[:, 1]

        report_text = classification_report(
            y_test,
            y_pred,
            digits=4,
            zero_division=0,
        )
        print(report_text)

        auc = roc_auc_score(y_test, y_prob)
        return {"auc": float(auc)}

    # ---------------------------
    # Inference / Persistence
    # ---------------------------

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
    # No f-string placeholders needed
    print("Training complete.")
