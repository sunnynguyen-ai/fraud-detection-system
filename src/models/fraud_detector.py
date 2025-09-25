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

import pandas as pd
import numpy as np
import joblib
import json
import os
from datetime import datetime
from typing import Dict, List, Tuple, Any

# ML Libraries
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, 
    precision_recall_curve, roc_curve, f1_score, precision_score, recall_score
)
from sklearn.utils.class_weight import compute_class_weight

# Advanced ML
import xgboost as xgb
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

# Explainability
import shap

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Suppress warnings
import warnings
warnings.filterwarnings('ignore')


class BaseDetector:
    """
    Base class for fraud detection models.

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
        balance_data: bool = True
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
        
        if balance_data:
            print(f"📊 Original training data: {y_train.value_counts().to_dict()}")
            smote = SMOTE(random_state=self.random_state)
            X_train, y_train = smote.fit_resample(X_train, y_train)
            print(f"⚖️ Balanced training data: {pd.Series(y_train).value_counts().to_dict()}")

        return X_train, X_test, y_train, y_test
    
    def evaluate_model(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
        """
        Evaluate model on test data.

        Args:
            X_test (pd.DataFrame): Test features.
            y_test (pd.Series): Test targets.

        Returns:
            dict: Evaluation metrics (accuracy, precision, recall, f1_score, roc_auc).
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before evaluation!")

        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        
        metrics = {
            'accuracy': self.model.score(X_test, y_test),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_pred_proba)
        }
        self.training_metrics.update(metrics)
        return metrics
    
    def plot_performance(self, X_test: pd.DataFrame, y_test: pd.Series) -> None:
        """
        Plot confusion matrix, ROC curve, PR curve, and feature importances.

        Args:
            X_test (pd.DataFrame): Test features.
            y_test (pd.Series): Test targets.
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before plotting!")

        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'{self.model_name} Performance Analysis', fontsize=16)

        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0,0])
        axes[0,0].set_title('Confusion Matrix')
        axes[0,0].set_xlabel('Predicted')
        axes[0,0].set_ylabel('Actual')

        # ROC Curve
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        axes[0,1].plot(fpr, tpr, label=f'ROC (AUC = {roc_auc_score(y_test, y_pred_proba):.3f})')
        axes[0,1].plot([0, 1], [0, 1], 'k--')
        axes[0,1].set_title('ROC Curve')
        axes[0,1].legend()

        # Precision-Recall Curve
        precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
        axes[1,0].plot(recall, precision)
        axes[1,0].set_title('Precision-Recall Curve')
        axes[1,0].set_xlabel('Recall')
        axes[1,0].set_ylabel('Precision')

        # Feature Importance
        if hasattr(self.model, 'feature_importances_'):
            feature_importance = pd.DataFrame({
                'feature': self.feature_names,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False).head(10)
            axes[1,1].barh(feature_importance['feature'], feature_importance['importance'])
            axes[1,1].set_title('Top 10 Feature Importances')

        plt.tight_layout()
        plt.show()
    
    def save_model(self, filepath: str) -> None:
        """Save trained model to disk."""
        if not self.is_trained:
            raise ValueError("Model must be trained before saving!")

        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        model_data = {
            'model': self.model,
            'model_name': self.model_name,
            'feature_names': self.feature_names,
            'training_metrics': self.training_metrics,
            'is_trained': self.is_trained,
            'training_date': datetime.now().isoformat()
        }
        joblib.dump(model_data, filepath)
        print(f"✅ Model saved to {filepath}")
    
    def load_model(self, filepath: str) -> None:
        """Load trained model from disk."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")

        model_data = joblib.load(filepath)
        self.model = model_data['model']
        self.model_name = model_data['model_name']
        self.feature_names = model_data['feature_names']
        self.training_metrics = model_data['training_metrics']
        self.is_trained = model_data['is_trained']
        print(f"✅ Model loaded from {filepath}")
