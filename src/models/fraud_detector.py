"""
Fraud Detection Model Classes

This module provides production-ready machine learning models for fraud detection:
- Individual model classes (Random Forest, XGBoost, Logistic Regression)
- Ensemble model combining multiple algorithms
- Model evaluation and performance tracking
- SHAP explainability integration
- Model persistence and loading

Author: Sunny Nguyen
"""

import json
import os
# Suppress warnings
import warnings
from datetime import datetime
from typing import Any, Dict, List, Tuple

import joblib
# Visualization
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
# Explainability
import shap
# Advanced ML
import xgboost as xgb
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
# ML Libraries
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (classification_report, confusion_matrix, f1_score,
                             precision_recall_curve, precision_score,
                             recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import (GridSearchCV, cross_val_score,
                                     train_test_split)
from sklearn.utils.class_weight import compute_class_weight

warnings.filterwarnings("ignore")


class BaseDetector:
    """
    Base class for fraud detection models

    Provides common functionality for all fraud detection models including
    training, evaluation, and prediction capabilities.
    """

    def __init__(self, model_name: str, random_state: int = 42):
        """
        Initialize base detector.

        Args:
            model_name (str): Name of the model.
            random_state (int): Random state for reproducibility.
        """
        self.model_name = model_name
        self.random_state = random_state
        self.model = None
        self.is_trained = False
        self.training_metrics: Dict[str, Any] = {}
        self.feature_names: List[str] = []

    def prepare_data(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        test_size: float = 0.2,
        balance_data: bool = True,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Prepare training and testing data with optional balancing.

        Args:
            X (pd.DataFrame): Feature matrix.
            y (pd.Series): Target variable.
            test_size (float, optional): Proportion of data for testing. Defaults to 0.2.
            balance_data (bool, optional): Whether to balance the dataset. Defaults to True.

        Returns:
            tuple: (X_train, X_test, y_train, y_test).
        """
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state, stratify=y
        )

        # Balance training data if requested
        if balance_data:
            print(f"  📊 Original training data: {y_train.value_counts().to_dict()}")

            # Use SMOTE for oversampling minority class
            smote = SMOTE(random_state=self.random_state)
            X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

            print(
                f"  ⚖️ Balanced training data: {pd.Series(y_train_balanced).value_counts().to_dict()}"
            )

            return X_train_balanced, X_test, y_train_balanced, y_test

        return X_train, X_test, y_train, y_test

    def evaluate_model(
        self, X_test: pd.DataFrame, y_test: pd.Series
    ) -> Dict[str, float]:
        """
        Comprehensive model evaluation

        Args:
            X_test (pd.DataFrame): Test features
            y_test (pd.Series): Test targets

        Returns:
            dict: Evaluation metrics (accuracy, precision, recall, f1_score, roc_auc).
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before evaluation!")

        # Predictions
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]

        # Calculate metrics
        metrics = {
            "accuracy": self.model.score(X_test, y_test),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1_score": f1_score(y_test, y_pred),
            "roc_auc": roc_auc_score(y_test, y_pred_proba),
        }

        # Store metrics
        self.training_metrics.update(metrics)

        return metrics

    def plot_performance(self, X_test: pd.DataFrame, y_test: pd.Series) -> None:
        """
        Plot model performance visualizations

        Args:
            X_test (pd.DataFrame): Test features
            y_test (pd.Series): Test targets
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before plotting!")

        # Predictions
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]

        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f"{self.model_name} Performance Analysis", fontsize=16)

        # 1. Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[0, 0])
        axes[0, 0].set_title("Confusion Matrix")
        axes[0, 0].set_xlabel("Predicted")
        axes[0, 0].set_ylabel("Actual")

        # 2. ROC Curve
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        axes[0, 1].plot(
            fpr, tpr, label=f"ROC (AUC = {roc_auc_score(y_test, y_pred_proba):.3f})"
        )
        axes[0, 1].plot([0, 1], [0, 1], "k--")
        axes[0, 1].set_xlabel("False Positive Rate")
        axes[0, 1].set_ylabel("True Positive Rate")
        axes[0, 1].set_title("ROC Curve")
        axes[0, 1].legend()

        # 3. Precision-Recall Curve
        precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
        axes[1, 0].plot(recall, precision)
        axes[1, 0].set_xlabel("Recall")
        axes[1, 0].set_ylabel("Precision")
        axes[1, 0].set_title("Precision-Recall Curve")

        # 4. Feature Importance (if available)
        if hasattr(self.model, "feature_importances_"):
            feature_importance = (
                pd.DataFrame(
                    {
                        "feature": self.feature_names
                        or [
                            f"feature_{i}"
                            for i in range(len(self.model.feature_importances_))
                        ],
                        "importance": self.model.feature_importances_,
                    }
                )
                .sort_values("importance", ascending=False)
                .head(10)
            )
            axes[1, 1].barh(
                feature_importance["feature"], feature_importance["importance"]
            )
            axes[1, 1].set_title("Top 10 Feature Importances")
            axes[1, 1].set_xlabel("Importance")

        plt.tight_layout()
        plt.show()

    def save_model(self, filepath: str) -> None:
        """Save trained model to disk"""
        if not self.is_trained:
            raise ValueError("Model must be trained before saving!")

        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        # Save model and metadata
        model_data = {
            "model": self.model,
            "model_name": self.model_name,
            "feature_names": self.feature_names,
            "training_metrics": self.training_metrics,
            "is_trained": self.is_trained,
            "training_date": datetime.now().isoformat(),
        }

        joblib.dump(model_data, filepath)
        print(f"✅ Model saved to {filepath}")

    def load_model(self, filepath: str) -> None:
        """Load trained model from disk"""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")

        model_data = joblib.load(filepath)

        self.model = model_data["model"]
        self.model_name = model_data["model_name"]
        self.feature_names = model_data["feature_names"]
        self.training_metrics = model_data["training_metrics"]
        self.is_trained = model_data["is_trained"]

        print(f"✅ Model loaded from {filepath}")


class RandomForestDetector(BaseDetector):
    """
    Random Forest-based fraud detector

    Excellent for handling feature interactions and providing
    robust predictions with feature importance insights.
    """

    def __init__(self, random_state: int = 42, **kwargs):
        """
        Initialize Random Forest detector

        Args:
            random_state (int): Random state for reproducibility
            **kwargs: Additional parameters for RandomForestClassifier
        """
        super().__init__("Random Forest", random_state)

        # Default parameters optimized for fraud detection
        self.params = {
            "n_estimators": 100,
            "max_depth": 10,
            "min_samples_split": 10,
            "min_samples_leaf": 5,
            "class_weight": "balanced",
            "random_state": random_state,
            **kwargs,
        }

        self.model = RandomForestClassifier(**self.params)

    def train(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        tune_hyperparameters: bool = False,
        cv_folds: int = 5,
    ) -> None:
        """
        Train Random Forest model

        Args:
            X (pd.DataFrame): Training features
            y (pd.Series): Training targets
            tune_hyperparameters (bool): Whether to perform hyperparameter tuning
            cv_folds (int): Number of cross-validation folds
        """
        self.feature_names = X.columns.tolist()

        print(f"🌲 Training {self.model_name}...")

        if tune_hyperparameters:
            print("  🔧 Tuning hyperparameters...")
            param_grid = {
                "n_estimators": [50, 100, 200],
                "max_depth": [5, 10, 15, None],
                "min_samples_split": [5, 10, 20],
                "min_samples_leaf": [2, 5, 10],
            }

            grid_search = GridSearchCV(
                RandomForestClassifier(
                    class_weight="balanced", random_state=self.random_state
                ),
                param_grid,
                cv=cv_folds,
                scoring="f1",
                n_jobs=-1,
            )
            grid_search.fit(X, y)

            self.model = grid_search.best_estimator_
            self.params = grid_search.best_params_
            print(f"  ✅ Best parameters: {self.params}")
        else:
            self.model.fit(X, y)

        # Cross-validation score
        cv_scores = cross_val_score(self.model, X, y, cv=cv_folds, scoring="f1")
        self.training_metrics["cv_f1_mean"] = cv_scores.mean()
        self.training_metrics["cv_f1_std"] = cv_scores.std()

        self.is_trained = True
        print(
            f"  ✅ Training completed! CV F1: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}"
        )


class XGBoostDetector(BaseDetector):
    """
    XGBoost-based fraud detector

    Gradient boosting model excellent for capturing complex patterns
    and handling imbalanced datasets effectively.
    """

    def __init__(self, random_state: int = 42, **kwargs):
        """
        Initialize XGBoost detector

        Args:
            random_state (int): Random state for reproducibility
            **kwargs: Additional parameters for XGBClassifier
        """
        super().__init__("XGBoost", random_state)

        # Default parameters optimized for fraud detection
        self.params = {
            "n_estimators": 100,
            "max_depth": 6,
            "learning_rate": 0.1,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "scale_pos_weight": 99,  # For imbalanced data (99:1 ratio)
            "random_state": random_state,
            "eval_metric": "logloss",
            **kwargs,
        }

        self.model = xgb.XGBClassifier(**self.params)

    def train(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        tune_hyperparameters: bool = False,
        cv_folds: int = 5,
    ) -> None:
        """
        Train XGBoost model

        Args:
            X (pd.DataFrame): Training features
            y (pd.Series): Training targets
            tune_hyperparameters (bool): Whether to perform hyperparameter tuning
            cv_folds (int): Number of cross-validation folds
        """
        self.feature_names = X.columns.tolist()

        print(f"🚀 Training {self.model_name}...")

        if tune_hyperparameters:
            print("  🔧 Tuning hyperparameters...")
            param_grid = {
                "n_estimators": [50, 100, 200],
                "max_depth": [3, 6, 10],
                "learning_rate": [0.01, 0.1, 0.2],
                "subsample": [0.8, 0.9, 1.0],
            }

            grid_search = GridSearchCV(
                xgb.XGBClassifier(
                    random_state=self.random_state, eval_metric="logloss"
                ),
                param_grid,
                cv=cv_folds,
                scoring="f1",
                n_jobs=-1,
            )
            grid_search.fit(X, y)

            self.model = grid_search.best_estimator_
            self.params = grid_search.best_params_
            print(f"  ✅ Best parameters: {self.params}")
        else:
            self.model.fit(X, y)

        # Cross-validation score
        cv_scores = cross_val_score(self.model, X, y, cv=cv_folds, scoring="f1")
        self.training_metrics["cv_f1_mean"] = cv_scores.mean()
        self.training_metrics["cv_f1_std"] = cv_scores.std()

        self.is_trained = True
        print(
            f"  ✅ Training completed! CV F1: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}"
        )


class LogisticRegressionDetector(BaseDetector):
    """
    Logistic Regression-based fraud detector

    Provides interpretable baseline predictions and probability calibration.
    Excellent for understanding linear relationships.
    """

    def __init__(self, random_state: int = 42, **kwargs):
        """
        Initialize Logistic Regression detector

        Args:
            random_state (int): Random state for reproducibility
            **kwargs: Additional parameters for LogisticRegression
        """
        super().__init__("Logistic Regression", random_state)

        # Default parameters
        self.params = {
            "class_weight": "balanced",
            "random_state": random_state,
            "max_iter": 1000,
            "solver": "liblinear",
            **kwargs,
        }

        self.model = LogisticRegression(**self.params)

    def train(self, X: pd.DataFrame, y: pd.Series, cv_folds: int = 5) -> None:
        """
        Train Logistic Regression model

        Args:
            X (pd.DataFrame): Training features
            y (pd.Series): Training targets
            cv_folds (int): Number of cross-validation folds
        """
        self.feature_names = X.columns.tolist()

        print(f"📈 Training {self.model_name}...")

        self.model.fit(X, y)

        # Cross-validation score
        cv_scores = cross_val_score(self.model, X, y, cv=cv_folds, scoring="f1")
        self.training_metrics["cv_f1_mean"] = cv_scores.mean()
        self.training_metrics["cv_f1_std"] = cv_scores.std()

        self.is_trained = True
        print(
            f"  ✅ Training completed! CV F1: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}"
        )


class EnsembleFraudDetector:
    """
    Ensemble fraud detector combining multiple models

    Combines Random Forest, XGBoost, and Logistic Regression
    using weighted voting for robust predictions.
    """

    def __init__(self, weights: Dict[str, float] = None, random_state: int = 42):
        """
        Initialize ensemble detector

        Args:
            weights (dict): Weights for each model in ensemble
            random_state (int): Random state for reproducibility
        """
        self.random_state = random_state

        # Default weights optimized for fraud detection
        self.weights = weights or {
            "random_forest": 0.4,
            "xgboost": 0.5,
            "logistic_regression": 0.1,
        }

        # Initialize individual models
        self.models = {
            "random_forest": RandomForestDetector(random_state=random_state),
            "xgboost": XGBoostDetector(random_state=random_state),
            "logistic_regression": LogisticRegressionDetector(
                random_state=random_state
            ),
        }

        self.is_trained = False
        self.feature_names: List[str] = []
        self.ensemble_metrics: Dict[str, Any] = {}
        self.explainer = None

    def train(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        tune_hyperparameters: bool = False,
        test_size: float = 0.2,
        balance_data: bool = True,
    ) -> None:
        """
        Train all models in the ensemble

        Args:
            X (pd.DataFrame): Training features
            y (pd.Series): Training targets
            tune_hyperparameters (bool): Whether to tune hyperparameters
            test_size (float): Test set size
            balance_data (bool): Whether to balance training data
        """
        self.feature_names = X.columns.tolist()

        print("🎯 Training Ensemble Fraud Detector...")
        print(f"📊 Dataset: {X.shape[0]} samples, {X.shape[1]} features")
        print(f"⚖️ Class distribution: {dict(y.value_counts())}")

        # Prepare data
        X_train, X_test, y_train, y_test = self.models["random_forest"].prepare_data(
            X, y, test_size=test_size, balance_data=balance_data
        )

        # Train individual models
        print("\n🔄 Training individual models...")

        # Random Forest
        self.models["random_forest"].train(X_train, y_train, tune_hyperparameters)

        # XGBoost
        self.models["xgboost"].train(X_train, y_train, tune_hyperparameters)

        # Logistic Regression
        self.models["logistic_regression"].train(X_train, y_train)

        # Evaluate ensemble
        print("\n📈 Evaluating ensemble performance...")
        ensemble_metrics = self.evaluate_ensemble(X_test, y_test)
        self.ensemble_metrics = ensemble_metrics

        # Initialize SHAP explainer with the best performing model
        best_model_name = max(
            self.models.keys(),
            key=lambda k: self.models[k].training_metrics.get("f1_score", 0),
        )
        best_model = self.models[best_model_name].model

        try:
            self.explainer = shap.Explainer(
                best_model, X_train.sample(min(100, len(X_train)))
            )
        except Exception as e:
            print(f"⚠️ Could not initialize SHAP explainer: {e}")

        self.is_trained = True
        print(f"\n✅ Ensemble training completed!")
        print(f"🎯 Ensemble F1 Score: {ensemble_metrics['f1_score']:.4f}")
        print(f"🎯 Ensemble ROC-AUC: {ensemble_metrics['roc_auc']:.4f}")

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict fraud probabilities for new data."""
        if not self.is_trained:
            raise ValueError("Ensemble must be trained before making predictions!")

        # Get predictions from each model
        predictions = []
        for model_name, model in self.models.items():
            pred_proba = model.model.predict_proba(X)[:, 1]
            weighted_pred = pred_proba * self.weights[model_name]
            predictions.append(weighted_pred)

        # Combine predictions
        ensemble_pred = np.sum(predictions, axis=0)

        # Convert to probability format [prob_class_0, prob_class_1]
        prob_class_1 = ensemble_pred
        prob_class_0 = 1 - prob_class_1

        return np.column_stack([prob_class_0, prob_class_1])

    def predict(self, X: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        """Return binary predictions from ensemble probabilities."""
        proba = self.predict_proba(X)
        return (proba[:, 1] >= threshold).astype(int)

    def explain_prediction(self, X: pd.DataFrame, index: int = 0) -> Dict[str, Any]:
        """
        Explain a single prediction using SHAP values.

        Args:
            X (pd.DataFrame): Input features.
            index (int): Index of sample to explain.

        Returns:
            Dict[str, Any]: Explanation payload including top feature contributions.
        """
        if not self.is_trained:
            raise ValueError("Ensemble must be trained before explaining predictions!")

        if self.explainer is None:
            return {"error": "SHAP explainer not available"}

        # Get prediction
        sample = X.iloc[index : index + 1]
        prediction = self.predict_proba(sample)[0, 1]

        try:
            # Get SHAP values
            shap_values = self.explainer(sample)

            # Get top contributing features
            feature_contributions = []
            for feature, shap_val in zip(self.feature_names, shap_values.values[0]):
                feature_contributions.append(
                    {
                        "feature": feature,
                        "value": sample[feature].iloc[0],
                        "shap_value": float(shap_val),
                        "contribution": (
                            "increases fraud risk"
                            if shap_val > 0
                            else "decreases fraud risk"
                        ),
                    }
                )

            # Sort by absolute SHAP value
            feature_contributions.sort(key=lambda x: abs(x["shap_value"]), reverse=True)

            return {
                "prediction_probability": float(prediction),
                "risk_level": (
                    "HIGH"
                    if prediction > 0.8
                    else "MEDIUM" if prediction > 0.3 else "LOW"
                ),
                "top_features": feature_contributions[:5],
                "all_features": feature_contributions,
            }

        except Exception as e:
            return {
                "prediction_probability": float(prediction),
                "risk_level": (
                    "HIGH"
                    if prediction > 0.8
                    else "MEDIUM" if prediction > 0.3 else "LOW"
                ),
                "error": f"Could not generate SHAP explanation: {e}",
            }

    def evaluate_ensemble(
        self, X_test: pd.DataFrame, y_test: pd.Series
    ) -> Dict[str, float]:
        """
        Evaluate ensemble performance.

        Args:
            X_test (pd.DataFrame): Test features.
            y_test (pd.Series): Test targets.

        Returns:
            dict: Performance metrics.
        """
        # Get ensemble predictions
        y_pred_proba = self.predict_proba(X_test)[:, 1]
        y_pred = self.predict(X_test)

        # Calculate metrics
        metrics = {
            "accuracy": np.mean(y_pred == y_test),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1_score": f1_score(y_test, y_pred),
            "roc_auc": roc_auc_score(y_test, y_pred_proba),
        }

        return metrics

    def save_ensemble(self, filepath: str) -> None:
        """Save entire ensemble to disk"""
        if not self.is_trained:
            raise ValueError("Ensemble must be trained before saving!")

        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        # Save ensemble data
        ensemble_data = {
            "models": self.models,
            "weights": self.weights,
            "feature_names": self.feature_names,
            "ensemble_metrics": self.ensemble_metrics,
            "is_trained": self.is_trained,
            "training_date": datetime.now().isoformat(),
        }

        joblib.dump(ensemble_data, filepath)
        print(f"✅ Ensemble saved to {filepath}")

    def load_ensemble(self, filepath: str) -> None:
        """Load entire ensemble from disk"""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Ensemble file not found: {filepath}")

        ensemble_data = joblib.load(filepath)

        self.models = ensemble_data["models"]
        self.weights = ensemble_data["weights"]
        self.feature_names = ensemble_data["feature_names"]
        self.ensemble_metrics = ensemble_data["ensemble_metrics"]
        self.is_trained = ensemble_data["is_trained"]

        print(f"✅ Ensemble loaded from {filepath}")


# Example usage and testing
def test_fraud_detection_models():
    """Test the fraud detection models with sample data"""

    # Import required modules
    import os
    import sys

    sys.path.append(os.path.dirname(os.path.dirname(__file__)))

    from data_processing.feature_engineering import AdvancedFeatureEngineering
    from data_processing.generate_data import create_fraud_dataset

    print("🔄 Creating and preprocessing sample dataset...")

    # Create sample data
    df = create_fraud_dataset(n_samples=5000)  # Smaller for quick testing

    # Feature engineering
    fe = AdvancedFeatureEngineering(target_column="Class")
    df_processed = fe.fit_transform(df)

    # Prepare features and target
    X = df_processed.drop("Class", axis=1)
    y = df_processed["Class"]

    print(f"📊 Processed dataset: {X.shape}")

    # Test individual models
    print("\n🧪 Testing individual models...")

    # Random Forest
    rf_detector = RandomForestDetector()
    X_train, X_test, y_train, y_test = rf_detector.prepare_data(
        X, y, balance_data=False
    )
    rf_detector.train(X_train, y_train)
    rf_metrics = rf_detector.evaluate_model(X_test, y_test)
    print(f"🌲 Random Forest F1: {rf_metrics['f1_score']:.4f}")

    # Test ensemble
    print("\n🎯 Testing ensemble model...")
    ensemble = EnsembleFraudDetector()
    ensemble.train(X, y, test_size=0.3, balance_data=False)

    # Test explanation
    print("\n🔍 Testing prediction explanation...")
    explanation = ensemble.explain_prediction(X.head(1), 0)
    print(f"Sample prediction: {explanation.get('prediction_probability', 'N/A'):.4f}")
    print(f"Risk level: {explanation.get('risk_level', 'N/A')}")

    return ensemble


if __name__ == "__main__":
    # Run test
    test_fraud_detection_models()
