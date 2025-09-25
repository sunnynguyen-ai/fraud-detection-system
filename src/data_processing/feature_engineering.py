"""
Feature Engineering Pipeline for Fraud Detection

This module provides comprehensive feature engineering capabilities including:
- Time-based feature extraction
- Statistical aggregations and rolling windows
- Advanced feature transformations
- Missing value handling strategies
- Feature scaling and encoding

Author: Sunny Nguyen
"""

import warnings
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import LabelEncoder, RobustScaler, StandardScaler

warnings.filterwarnings("ignore")


class AdvancedFeatureEngineering:
    """
    Advanced feature engineering pipeline for fraud detection.

    This class implements sophisticated feature engineering techniques
    commonly used in production fraud detection systems.
    """

    def __init__(self, target_column: str = "Class"):
        """
        Initialize the feature engineering pipeline.

        Args:
            target_column: Name of the target variable column.
        """
        self.target_column: str = target_column
        self.scalers: Dict[str, RobustScaler | StandardScaler] = {}
        self.encoders: Dict[str, LabelEncoder] = {}
        self.feature_stats: Dict[str, float] = {}
        self.selected_features: Optional[List[str]] = None
        self.is_fitted: bool = False

    def create_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract comprehensive time-based features from timestamp.

        Args:
            df: Input dataframe with 'Time' column.

        Returns:
            Dataframe with additional time features.
        """
        df = df.copy()

        if "Time" in df.columns:
            # Assuming Time is in seconds from some reference point
            df["hour"] = (df["Time"] % (24 * 3600)) // 3600
            df["day_of_week"] = (df["Time"] // (24 * 3600)) % 7
            df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

            # Business hours (9 AM to 5 PM)
            df["is_business_hours"] = (
                (df["hour"] >= 9) & (df["hour"] <= 17)
            ).astype(int)

            # Late night transactions (11 PM to 6 AM)
            df["is_late_night"] = ((df["hour"] >= 23) | (df["hour"] <= 6)).astype(int)

            # Peak hours (lunch: 12–14, evening: 18–20)
            df["is_peak_hours"] = (
                ((df["hour"] >= 12) & (df["hour"] <= 14))
                | ((df["hour"] >= 18) & (df["hour"] <= 20))
            ).astype(int)

        return df

    def create_amount_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create sophisticated amount-based features.

        Args:
            df: Input dataframe with 'Amount' column.

        Returns:
            Dataframe with additional amount features.
        """
        df = df.copy()

        if "Amount" in df.columns:
            # Log transformation for skewed amounts
            df["amount_log"] = np.log1p(df["Amount"])

            # Amount categories
            df["amount_category"] = pd.cut(
                df["Amount"],
                bins=[0, 10, 100, 1000, np.inf],
                labels=["small", "medium", "large", "very_large"],
            )

            # Z-score normalization for outlier detection
            amount_mean = df["Amount"].mean()
            amount_std = df["Amount"].std(ddof=0) or 1.0
            df["amount_zscore"] = (df["Amount"] - amount_mean) / amount_std
            df["is_amount_outlier"] = (np.abs(df["amount_zscore"]) > 3).astype(int)

            # Round number indicators (psychological pricing)
            df["is_round_amount"] = (df["Amount"] % 1 == 0).astype(int)
            df["is_round_10"] = (df["Amount"] % 10 == 0).astype(int)
            df["is_round_100"] = (df["Amount"] % 100 == 0).astype(int)

        return df

    def create_statistical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create statistical features from the V* columns.

        Args:
            df: Input dataframe with V1..Vn columns.

        Returns:
            Dataframe with additional statistical features.
        """
        df = df.copy()

        v_columns = [c for c in df.columns if c.startswith("V") and c[1:].isdigit()]

        if v_columns:
            v_data = df[v_columns]

            # Statistical aggregations
            df["v_mean"] = v_data.mean(axis=1)
            df["v_std"] = v_data.std(axis=1)
            df["v_skew"] = v_data.skew(axis=1)
            df["v_kurt"] = v_data.kurtosis(axis=1)
            df["v_median"] = v_data.median(axis=1)
            df["v_max"] = v_data.max(axis=1)
            df["v_min"] = v_data.min(axis=1)
            df["v_range"] = df["v_max"] - df["v_min"]

            # Quartiles and IQR
            df["v_q25"] = v_data.quantile(0.25, axis=1)
            df["v_q75"] = v_data.quantile(0.75, axis=1)
            df["v_iqr"] = df["v_q75"] - df["v_q25"]

            # Count of outliers in V features
            z = stats.zscore(v_data, axis=1, nan_policy="omit")
            v_outliers = np.abs(z) > 2
            df["v_outlier_count"] = np.nan_to_num(v_outliers).sum(axis=1)

            # Correlation-like ratios with amount (first few to limit cost)
            if "Amount" in df.columns:
                denom = df["Amount"] + 1e-8
                for col in v_columns[:5]:
                    df[f"{col}_amount_ratio"] = df[col] / denom

        return df

    def create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create interaction features between important variables.

        Args:
            df: Input dataframe.

        Returns:
            Dataframe with interaction features.
        """
        df = df.copy()

        # Time-Amount interactions
        if "hour" in df.columns and "Amount" in df.columns and "amount_log" in df.columns:
            df["hour_amount_interaction"] = df["hour"] * df["amount_log"]
            df["weekend_amount_interaction"] = df["is_weekend"] * df["amount_log"]
            df["business_hours_amount"] = df["is_business_hours"] * df["amount_log"]

        # V feature interactions (if at least V1..V4 exist)
        v_columns = [c for c in df.columns if c.startswith("V") and c[1:].isdigit()]
        if {"V1", "V2", "V3", "V4"}.issubset(set(v_columns)):
            df["v1_v2_interaction"] = df["V1"] * df["V2"]
            df["v3_v4_interaction"] = df["V3"] * df["V4"]
            df["v1_v3_ratio"] = df["V1"] / (np.abs(df["V3"]) + 1e-8)
            df["v2_v4_ratio"] = df["V2"] / (np.abs(df["V4"]) + 1e-8)

        return df

    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Intelligent missing value handling strategy.

        Args:
            df: Input dataframe.

        Returns:
            Dataframe with handled missing values.
        """
        df = df.copy()

        for column in df.columns:
            if df[column].isnull().any():
                if df[column].dtype == "object":
                    # Categorical columns: fill with 'Unknown'
                    df[column] = df[column].fillna("Unknown")
                else:
                    # Create missing indicator BEFORE filling
                    df[f"{column}_was_missing"] = df[column].isnull().astype(int)
                    # Numerical columns: median is robust to outliers
                    median_value = df[column].median()
                    df[column] = df[column].fillna(median_value)

        return df

    def encode_categorical_features(
        self, df: pd.DataFrame, fit: bool = True
    ) -> pd.DataFrame:
        """
        Encode categorical features using appropriate encoding strategies.

        Args:
            df: Input dataframe.
            fit: Whether to fit encoders or use existing ones.

        Returns:
            Dataframe with encoded categorical features.
        """
        df = df.copy()

        categorical_columns = df.select_dtypes(include=["object", "category"]).columns

        for column in categorical_columns:
            if column == self.target_column:
                continue

            if fit:
                encoder = LabelEncoder()
                df[column] = encoder.fit_transform(df[column].astype(str))
                self.encoders[column] = encoder
            else:
                if column in self.encoders:
                    enc = self.encoders[column]
                    df[column] = df[column].astype(str)
                    # Map unseen categories to a placeholder
                    known = set(enc.classes_)
                    unknown_mask = ~df[column].isin(known)
                    if unknown_mask.any():
                        placeholder = enc.classes_[0]
                        df.loc[unknown_mask, column] = placeholder
                    df[column] = enc.transform(df[column])

        return df

    def scale_features(
        self, df: pd.DataFrame, fit: bool = True, method: str = "robust"
    ) -> pd.DataFrame:
        """
        Scale numerical features using specified scaling method.

        Args:
            df: Input dataframe.
            fit: Whether to fit scalers or use existing ones.
            method: Scaling method ('standard', 'robust').

        Returns:
            Dataframe with scaled features.
        """
        df = df.copy()

        numerical_columns = df.select_dtypes(include=[np.number]).columns
        if self.target_column in numerical_columns:
            numerical_columns = numerical_columns.drop(self.target_column)

        if len(numerical_columns) == 0:
            return df

        if fit:
            scaler = StandardScaler() if method == "standard" else RobustScaler()
            df[numerical_columns] = scaler.fit_transform(df[numerical_columns])
            self.scalers["main"] = scaler
        else:
            if "main" in self.scalers:
                df[numerical_columns] = self.scalers["main"].transform(
                    df[numerical_columns]
                )

        return df

    def select_features(
        self, df: pd.DataFrame, target: pd.Series, k: int = 50, method: str = "f_classif"
    ) -> pd.DataFrame:
        """
        Select top k features using statistical tests.

        Args:
            df: Input dataframe.
            target: Target variable.
            k: Number of features to select.
            method: Feature selection method.

        Returns:
            Dataframe with selected features.
        """
        feature_columns = (
            df.columns.drop(self.target_column)
            if self.target_column in df.columns
            else df.columns
        )
        X = df[feature_columns]

        # Currently only f_classif; placeholder to extend later
        _ = method  # avoid unused-var warnings
        selector = SelectKBest(score_func=f_classif, k=min(k, len(feature_columns)))

        X_selected = selector.fit_transform(X, target)

        self.selected_features = X.columns[selector.get_support()].tolist()

        df_selected = pd.DataFrame(
            X_selected, columns=self.selected_features, index=df.index
        )

        if self.target_column in df.columns:
            df_selected[self.target_column] = df[self.target_column]

        return df_selected

    def fit_transform(
        self, df: pd.DataFrame, target_column: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Fit the feature engineering pipeline and transform the data.

        Args:
            df: Input dataframe.
            target_column: Target column name (optional).

        Returns:
            Transformed dataframe.
        """
        if target_column:
            self.target_column = target_column

        target = df[self.target_column] if self.target_column in df.columns else None

        print("🔧 Starting feature engineering pipeline...")
        print("  ⏰ Creating time features...")
        df = self.create_time_features(df)

        print("  💰 Creating amount features...")
        df = self.create_amount_features(df)

        print("  📊 Creating statistical features...")
        df = self.create_statistical_features(df)

        print("  🔗 Creating interaction features...")
        df = self.create_interaction_features(df)

        print("  🔧 Handling missing values...")
        df = self.handle_missing_values(df)

        print("  🏷️ Encoding categorical features...")
        df = self.encode_categorical_features(df, fit=True)

        print("  📏 Scaling features...")
        df = self.scale_features(df, fit=True)

        if target is not None:
            print("  🎯 Selecting best features...")
            df = self.select_features(df, target, k=50)

        self.is_fitted = True
        print("✅ Feature engineering pipeline completed!")
        print(f"📈 Final dataset shape: {df.shape}")

        return df

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform new data using fitted pipeline.

        Args:
            df: Input dataframe.

        Returns:
            Transformed dataframe.
        """
        if not self.is_fitted:
            raise ValueError("Pipeline must be fitted before transforming new data!")

        df = self.create_time_features(df)
        df = self.create_amount_features(df)
        df = self.create_statistical_features(df)
        df = self.create_interaction_features(df)
        df = self.handle_missing_values(df)
        df = self.encode_categorical_features(df, fit=False)
        df = self.scale_features(df, fit=False)

        if self.selected_features:
            keep = [f for f in self.selected_features if f in df.columns]
            df = df[keep]

        return df

    def get_feature_importance_summary(self) -> Dict[str, object]:
        """
        Get summary of feature engineering transformations.

        Returns:
            Summary of transformations applied.
        """
        return {
            "total_features_created": (
                len(self.selected_features) if self.selected_features else 0
            ),
            "selected_features": self.selected_features,
            "encoders_fitted": list(self.encoders.keys()),
            "scalers_fitted": list(self.scalers.keys()),
            "is_fitted": self.is_fitted,
        }


# Example usage / quick test (kept for convenience)
def test_feature_engineering():
    """Test the feature engineering pipeline with sample data."""
    import os
    import sys

    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    from data_processing.generate_data import create_fraud_dataset

    print("🔄 Creating sample dataset...")
    df = create_fraud_dataset(n_samples=10000)

    fe = AdvancedFeatureEngineering(target_column="Class")
    df_transformed = fe.fit_transform(df)

    print("\n📊 Transformation Results:")
    print(f"Original shape: {df.shape}")
    print(f"Transformed shape: {df_transformed.shape}")
    print(f"Features created: {df_transformed.shape[1] - df.shape[1]}")

    summary = fe.get_feature_importance_summary()
    print(f"\n🎯 Selected Features ({len(summary['selected_features'])}):")
    for i, feature in enumerate(summary["selected_features"][:10]):
        print(f"  {i + 1}. {feature}")
    if len(summary["selected_features"]) > 10:
        remaining = len(summary["selected_features"]) - 10
        print(f"  ... and {remaining} more")

    return df_transformed, fe


if __name__ == "__main__":
    test_feature_engineering()
