"""
Advanced Feature Engineering Pipeline for Fraud Detection
===========================================================

Production-ready feature engineering with:
- Velocity and frequency features for behavioral analysis
- Advanced statistical transformations
- Anomaly detection features
- Real-time feature computation capabilities
- Feature versioning and metadata tracking
- Memory-efficient processing for large datasets
- Feature store integration support

Author: Sunny Nguyen
Version: 2.0.0
"""

import hashlib
import json
import warnings
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Optional, Union

import numpy as np
import pandas as pd
from scipy import stats
# from scipy.special import boxcox1p  # (unused)
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.ensemble import IsolationForest
from sklearn.feature_selection import SelectKBest, chi2, f_classif, mutual_info_classif
from sklearn.preprocessing import (
    LabelEncoder,
    MinMaxScaler,
    PolynomialFeatures,
    PowerTransformer,
    QuantileTransformer,
    RobustScaler,
    StandardScaler,
)

warnings.filterwarnings("ignore")


@dataclass
class FeatureConfig:
    """Configuration for feature engineering pipeline"""

    # Feature creation flags
    enable_time_features: bool = True
    enable_amount_features: bool = True
    enable_statistical_features: bool = True
    enable_interaction_features: bool = True
    enable_velocity_features: bool = True
    enable_anomaly_features: bool = True
    enable_clustering_features: bool = True
    enable_polynomial_features: bool = False

    # Scaling configuration
    scaling_method: str = "robust"  # "standard", "robust", "minmax", "quantile", "power"

    # Feature selection
    feature_selection_method: str = "mutual_info"  # "f_classif", "chi2", "mutual_info"
    n_features_to_select: int = 50
    variance_threshold: float = 0.01

    # Dimensionality reduction
    enable_pca: bool = True
    pca_components: Union[int, float] = 0.95  # Keep 95% variance

    # Performance settings
    chunk_size: int = 10000  # For processing large datasets
    n_jobs: int = -1  # For parallel processing

    # Metadata tracking
    track_feature_importance: bool = True
    save_feature_metadata: bool = True


class FeatureStore:
    """Simple feature store for caching computed features"""

    def __init__(self):
        self.cache = {}
        self.metadata = {}

    def get_feature_hash(self, df: pd.DataFrame, feature_name: str) -> str:
        """Generate unique hash for feature based on input data"""
        data_hash = hashlib.md5(
            pd.util.hash_pandas_object(df).values.tobytes()
        ).hexdigest()
        return f"{feature_name}_{data_hash}"

    def store_features(
        self, feature_name: str, features: pd.DataFrame, metadata: Dict[str, Any] = None
    ):
        """Store computed features with metadata"""
        feature_hash = self.get_feature_hash(features, feature_name)
        self.cache[feature_hash] = features
        if metadata:
            self.metadata[feature_hash] = {
                **metadata,
                "created_at": datetime.now().isoformat(),
                "shape": features.shape,
            }

    def get_features(self, df: pd.DataFrame, feature_name: str) -> Optional[pd.DataFrame]:
        """Retrieve cached features if available"""
        feature_hash = self.get_feature_hash(df, feature_name)
        return self.cache.get(feature_hash)


class AdvancedFeatureEngineering:
    """
    Enhanced feature engineering pipeline for fraud detection

    This class implements state-of-the-art feature engineering techniques
    optimized for production fraud detection systems.
    """

    def __init__(self, config: Optional[FeatureConfig] = None, target_column: str = "Class"):
        """
        Initialize the feature engineering pipeline

        Args:
            config: Feature engineering configuration
            target_column: Name of the target variable column
        """
        self.config = config or FeatureConfig()
        self.target_column = target_column

        # Transformers and encoders
        self.scalers = {}
        self.encoders = {}
        self.selectors = {}
        self.dimensionality_reducers = {}

        # Feature metadata
        self.feature_stats = {}
        self.feature_importance = {}
        self.selected_features = None
        self.feature_creation_time = {}
        self.feature_versions = {}

        # State tracking
        self.is_fitted = False
        self.pipeline_version = "2.0.0"

        # Feature store for caching
        self.feature_store = FeatureStore()

        # Anomaly detectors
        self.anomaly_detectors = {}
        self.clustering_models = {}

    def create_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Enhanced time-based feature extraction with cyclical encoding

        Args:
            df: Input dataframe with 'Time' column

        Returns:
            DataFrame with additional time features
        """
        if not self.config.enable_time_features:
            return df

        df = df.copy()

        if "Time" in df.columns:
            # Basic time features
            df["hour"] = (df["Time"] % (24 * 3600)) // 3600
            df["day_of_week"] = (df["Time"] // (24 * 3600)) % 7
            df["day_of_month"] = (df["Time"] // (24 * 3600)) % 30
            df["week_of_month"] = df["day_of_month"] // 7

            # Cyclical encoding for periodic features
            df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
            df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
            df["day_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
            df["day_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)

            # Time-based flags
            df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
            df["is_business_hours"] = ((df["hour"] >= 9) & (df["hour"] <= 17)).astype(int)
            df["is_late_night"] = ((df["hour"] >= 23) | (df["hour"] <= 6)).astype(int)
            df["is_early_morning"] = ((df["hour"] >= 6) & (df["hour"] <= 9)).astype(int)
            df["is_lunch_time"] = ((df["hour"] >= 12) & (df["hour"] <= 14)).astype(int)
            df["is_evening"] = ((df["hour"] >= 18) & (df["hour"] <= 21)).astype(int)

            # Holiday indicators (simplified - would use holiday calendar in production)
            df["is_month_end"] = (df["day_of_month"] >= 28).astype(int)
            df["is_month_start"] = (df["day_of_month"] <= 3).astype(int)

            # Time since reference
            df["time_normalized"] = df["Time"] / df["Time"].max() if df["Time"].max() > 0 else 0

            self.feature_creation_time["time_features"] = datetime.now()

        return df

    def create_amount_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Advanced amount-based features with distribution analysis

        Args:
            df: Input dataframe with 'Amount' column

        Returns:
            DataFrame with additional amount features
        """
        if not self.config.enable_amount_features:
            return df

        df = df.copy()

        if "Amount" in df.columns:
            # Basic transformations
            df["amount_log"] = np.log1p(df["Amount"])
            df["amount_sqrt"] = np.sqrt(df["Amount"])
            df["amount_cbrt"] = np.cbrt(df["Amount"])

            # Box-Cox transformation for normality
            if (df["Amount"] > 0).all():
                df["amount_boxcox"], lambda_param = stats.boxcox(df["Amount"] + 1)
                self.feature_stats["amount_boxcox_lambda"] = lambda_param

            # Binning strategies
            df["amount_bin_quantile"] = pd.qcut(
                df["Amount"], q=10, labels=False, duplicates="drop"
            )
            df["amount_bin_log"] = pd.cut(np.log1p(df["Amount"]), bins=10, labels=False)

            # Statistical position
            amount_mean = df["Amount"].mean()
            amount_std = df["Amount"].std()
            amount_median = df["Amount"].median()

            df["amount_zscore"] = (df["Amount"] - amount_mean) / (amount_std + 1e-8)
            df["amount_mad_score"] = np.abs(df["Amount"] - amount_median) / (
                np.abs(df["Amount"] - amount_median).median() + 1e-8
            )

            # Outlier indicators
            df["is_amount_outlier_3sigma"] = (np.abs(df["amount_zscore"]) > 3).astype(int)
            df["is_amount_outlier_iqr"] = self._detect_outliers_iqr(df["Amount"])

            # Amount patterns
            df["is_round_amount"] = (df["Amount"] % 1 == 0).astype(int)
            df["is_round_10"] = (df["Amount"] % 10 == 0).astype(int)
            df["is_round_100"] = (df["Amount"] % 100 == 0).astype(int)
            df["amount_decimal_places"] = df["Amount"].apply(
                lambda x: len(str(x).split(".")[-1]) if "." in str(x) else 0
            )

            # Benford's Law features (first digit distribution)
            df["amount_first_digit"] = df["Amount"].apply(
                lambda x: int(str(x).replace(".", "")[0]) if x > 0 else 0
            )
            df["amount_second_digit"] = df["Amount"].apply(
                lambda x: int(str(x).replace(".", "")[1])
                if len(str(x).replace(".", "")) > 1
                else 0
            )

            # Relative amount features
            df["amount_to_mean_ratio"] = df["Amount"] / (amount_mean + 1e-8)
            df["amount_to_median_ratio"] = df["Amount"] / (amount_median + 1e-8)

            self.feature_creation_time["amount_features"] = datetime.now()

        return df

    def create_velocity_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create velocity and frequency features for behavioral analysis

        Args:
            df: Input dataframe

        Returns:
            DataFrame with velocity features
        """
        if not self.config.enable_velocity_features:
            return df

        df = df.copy()

        if "Time" in df.columns and "Amount" in df.columns:
            # Sort by time for sequential analysis
            df_sorted = df.sort_values("Time")

            # Time differences
            df_sorted["time_since_last"] = df_sorted["Time"].diff()
            df_sorted["time_since_last_fillna"] = df_sorted["time_since_last"].fillna(
                df_sorted["time_since_last"].median()
            )

            # Transaction velocity (rolling windows)
            for window in [10, 30, 100]:
                df_sorted[f"velocity_{window}"] = (
                    df_sorted["Amount"].rolling(window, min_periods=1).sum()
                    / (
                        df_sorted["time_since_last_fillna"].rolling(window, min_periods=1).sum()
                        + 1e-8
                    )
                )

                df_sorted[f"tx_count_{window}"] = df_sorted["Amount"].rolling(
                    window, min_periods=1
                ).count()

                df_sorted[f"amount_mean_{window}"] = df_sorted["Amount"].rolling(
                    window, min_periods=1
                ).mean()

                df_sorted[f"amount_std_{window}"] = df_sorted["Amount"].rolling(
                    window, min_periods=1
                ).std()

            # Acceleration (change in velocity)
            df_sorted["velocity_change"] = df_sorted["velocity_10"].diff()
            df_sorted["is_velocity_increasing"] = (df_sorted["velocity_change"] > 0).astype(
                int
            )

            # Frequency features
            df_sorted["tx_frequency_hour"] = 1 / (
                df_sorted["time_since_last_fillna"] / 3600 + 1e-8
            )

            # Return to original order
            df = df_sorted.sort_index()

            self.feature_creation_time["velocity_features"] = datetime.now()

        return df

    def create_statistical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Enhanced statistical features from V1-V28 columns

        Args:
            df: Input dataframe with V columns

        Returns:
            DataFrame with statistical features
        """
        if not self.config.enable_statistical_features:
            return df

        df = df.copy()

        # Get V columns
        v_columns = [col for col in df.columns if col.startswith("V") and col[1:].isdigit()]

        if v_columns:
            v_data = df[v_columns]

            # Basic statistics
            df["v_mean"] = v_data.mean(axis=1)
            df["v_std"] = v_data.std(axis=1)
            df["v_skew"] = v_data.skew(axis=1)
            df["v_kurt"] = v_data.kurtosis(axis=1)
            df["v_median"] = v_data.median(axis=1)
            df["v_mad"] = np.abs(
                v_data - v_data.median(axis=1).values.reshape(-1, 1)
            ).median(axis=1)

            # Range and quartiles
            df["v_max"] = v_data.max(axis=1)
            df["v_min"] = v_data.min(axis=1)
            df["v_range"] = df["v_max"] - df["v_min"]
            df["v_q25"] = v_data.quantile(0.25, axis=1)
            df["v_q75"] = v_data.quantile(0.75, axis=1)
            df["v_iqr"] = df["v_q75"] - df["v_q25"]

            # Higher moments
            df["v_moment_3"] = (
                (v_data - v_data.mean(axis=1).values.reshape(-1, 1)) ** 3
            ).mean(axis=1)
            df["v_moment_4"] = (
                (v_data - v_data.mean(axis=1).values.reshape(-1, 1)) ** 4
            ).mean(axis=1)

            # Entropy and energy
            df["v_entropy"] = -(v_data * np.log(np.abs(v_data) + 1e-8)).sum(axis=1)
            df["v_energy"] = (v_data ** 2).sum(axis=1)
            df["v_l1_norm"] = np.abs(v_data).sum(axis=1)
            df["v_l2_norm"] = np.sqrt((v_data ** 2).sum(axis=1))
            df["v_linf_norm"] = np.abs(v_data).max(axis=1)

            # Zero crossing and sign changes
            df["v_zero_crossings"] = (np.diff(np.sign(v_data), axis=1) != 0).sum(axis=1)
            df["v_positive_ratio"] = (v_data > 0).sum(axis=1) / len(v_columns)
            df["v_negative_ratio"] = (v_data < 0).sum(axis=1) / len(v_columns)

            # Outlier counts
            v_zscores = np.abs(stats.zscore(v_data, axis=1, nan_policy="omit"))
            df["v_outliers_2sigma"] = (v_zscores > 2).sum(axis=1)
            df["v_outliers_3sigma"] = (v_zscores > 3).sum(axis=1)

            # Correlation features
            if "Amount" in df.columns:
                for i, col in enumerate(v_columns[:5]):  # Top 5 for efficiency
                    df[f"{col}_amount_corr"] = v_data[col] * df["Amount"]
                    df[f"{col}_amount_ratio"] = v_data[col] / (df["Amount"] + 1e-8)

            # Principal component features
            if len(v_columns) > 3:
                try:
                    pca = PCA(n_components=3, random_state=42)
                    pca_features = pca.fit_transform(v_data)
                    df["v_pca_1"] = pca_features[:, 0]
                    df["v_pca_2"] = pca_features[:, 1]
                    df["v_pca_3"] = pca_features[:, 2]
                    df["v_pca_explained_var"] = pca.explained_variance_ratio_.sum()
                except Exception:
                    pass

            self.feature_creation_time["statistical_features"] = datetime.now()

        return df

    def create_anomaly_features(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """
        Create anomaly detection features using unsupervised methods

        Args:
            df: Input dataframe
            fit: Whether to fit new anomaly detectors

        Returns:
            DataFrame with anomaly scores
        """
        if not self.config.enable_anomaly_features:
            return df

        df = df.copy()

        # Select numerical features for anomaly detection
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        if self.target_column in numerical_cols:
            numerical_cols = numerical_cols.drop(self.target_column)

        if len(numerical_cols) > 0:
            X = df[numerical_cols].fillna(0)

            # Isolation Forest
            if fit:
                iso_forest = IsolationForest(
                    contamination=0.1, random_state=42, n_estimators=100
                )
                df["anomaly_score_iforest"] = iso_forest.fit_predict(X)
                df["anomaly_score_iforest_raw"] = iso_forest.score_samples(X)
                self.anomaly_detectors["isolation_forest"] = iso_forest
            elif "isolation_forest" in self.anomaly_detectors:
                iso_forest = self.anomaly_detectors["isolation_forest"]
                df["anomaly_score_iforest"] = iso_forest.predict(X)
                df["anomaly_score_iforest_raw"] = iso_forest.score_samples(X)

            # Local Outlier Factor (approximation for speed)
            # In production, use sklearn's LocalOutlierFactor
            df["anomaly_mahalanobis"] = self._mahalanobis_distance(X)

            self.feature_creation_time["anomaly_features"] = datetime.now()

        return df

    def create_clustering_features(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """
        Create clustering-based features for pattern recognition

        Args:
            df: Input dataframe
            fit: Whether to fit new clustering models

        Returns:
            DataFrame with cluster assignments and distances
        """
        if not self.config.enable_clustering_features:
            return df

        df = df.copy()

        # Select features for clustering
        feature_cols = [col for col in df.columns if col.startswith("V") and col[1:].isdigit()]

        if len(feature_cols) > 0:
            X = df[feature_cols].fillna(0)

            # K-Means clustering
            if fit:
                kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
                df["cluster_kmeans"] = kmeans.fit_predict(X)
                df["cluster_kmeans_dist"] = kmeans.transform(X).min(axis=1)
                self.clustering_models["kmeans"] = kmeans
            elif "kmeans" in self.clustering_models:
                kmeans = self.clustering_models["kmeans"]
                df["cluster_kmeans"] = kmeans.predict(X)
                df["cluster_kmeans_dist"] = kmeans.transform(X).min(axis=1)

            # DBSCAN for density-based clustering (only in fit mode)
            if fit and len(df) < 50000:  # Limit for performance
                dbscan = DBSCAN(eps=0.5, min_samples=5)
                df["cluster_dbscan"] = dbscan.fit_predict(X)
                df["is_dbscan_outlier"] = (df["cluster_dbscan"] == -1).astype(int)

            self.feature_creation_time["clustering_features"] = datetime.now()

        return df

    def create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create sophisticated interaction features between variables

        Args:
            df: Input dataframe

        Returns:
            DataFrame with interaction features
        """
        if not self.config.enable_interaction_features:
            return df

        df = df.copy()

        # Time-Amount interactions
        if "hour" in df.columns and "Amount" in df.columns:
            df["hour_amount_interaction"] = df["hour"] * np.log1p(df["Amount"])
            df["weekend_amount_interaction"] = df["is_weekend"] * np.log1p(df["Amount"])
            df["late_night_amount"] = df["is_late_night"] * np.log1p(df["Amount"])

            # Conditional features
            df["high_amount_late_night"] = (
                (df["is_late_night"] == 1) & (df["Amount"] > df["Amount"].quantile(0.9))
            ).astype(int)

            df["low_amount_business_hours"] = (
                (df["is_business_hours"] == 1) & (df["Amount"] < df["Amount"].quantile(0.1))
            ).astype(int)

        # V feature interactions
        v_columns = [col for col in df.columns if col.startswith("V") and col[1:].isdigit()]

        if len(v_columns) >= 4:
            # Top V feature interactions
            important_v = v_columns[:6]  # Use top 6 V features

            for i in range(len(important_v)):
                for j in range(i + 1, min(i + 3, len(important_v))):  # Limit interactions
                    col1, col2 = important_v[i], important_v[j]

                    # Multiplication
                    df[f"{col1}_{col2}_mult"] = df[col1] * df[col2]

                    # Division (safe)
                    df[f"{col1}_{col2}_ratio"] = df[col1] / (df[col2].abs() + 1e-8)

                    # Difference
                    df[f"{col1}_{col2}_diff"] = df[col1] - df[col2]

        # Polynomial features (if enabled)
        if self.config.enable_polynomial_features and len(v_columns) > 0:
            poly = PolynomialFeatures(degree=2, include_bias=False, interaction_only=True)
            poly_features = poly.fit_transform(df[v_columns[:3]])  # Use only first 3 V features
            poly_df = pd.DataFrame(
                poly_features[:, len(v_columns[:3]) :],  # Exclude original features
                columns=[f"poly_{i}" for i in range(poly_features.shape[1] - len(v_columns[:3]))],
                index=df.index,
            )
            df = pd.concat([df, poly_df.iloc[:, :10]], axis=1)  # Add only first 10 poly features

        self.feature_creation_time["interaction_features"] = datetime.now()

        return df

    def handle_missing_values(self, df: pd.DataFrame, strategy: str = "smart") -> pd.DataFrame:
        """
        Advanced missing value handling with multiple strategies

        Args:
            df: Input dataframe
            strategy: Imputation strategy ("smart", "median", "knn", "iterative")

        Returns:
            DataFrame with handled missing values
        """
        df = df.copy()

        for column in df.columns:
            if df[column].isnull().any():
                missing_ratio = df[column].isnull().sum() / len(df)

                # Create missing indicator for high missing ratio
                if missing_ratio > 0.05:
                    df[f"{column}_was_missing"] = df[column].isnull().astype(int)

                if df[column].dtype == "object":
                    # Categorical: mode or "Unknown"
                    if missing_ratio < 0.5:
                        df[column] = df[column].fillna(
                            df[column].mode()[0] if not df[column].mode().empty else "Unknown"
                        )
                    else:
                        df[column] = df[column].fillna("Unknown")
                else:
                    # Numerical imputation
                    if strategy == "smart":
                        if missing_ratio < 0.1:
                            # Low missing: use median
                            df[column] = df[column].fillna(df[column].median())
                        elif missing_ratio < 0.3:
                            # Moderate missing: use grouped median if possible
                            if "hour" in df.columns:
                                df[column] = df.groupby("hour")[column].transform(
                                    lambda x: x.fillna(x.median())
                                )
                            else:
                                df[column] = df[column].fillna(df[column].median())
                        else:
                            # High missing: use zero or flag
                            df[column] = df[column].fillna(0)
                    else:
                        df[column] = df[column].fillna(df[column].median())

        return df

    def encode_categorical_features(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """
        Enhanced categorical encoding with multiple strategies

        Args:
            df: Input dataframe
            fit: Whether to fit encoders

        Returns:
            DataFrame with encoded features
        """
        df = df.copy()
        categorical_columns = df.select_dtypes(include=["object", "category"]).columns

        for column in categorical_columns:
            if column != self.target_column:
                unique_count = df[column].nunique()

                if unique_count <= 2:
                    # Binary encoding
                    if fit:
                        encoder = LabelEncoder()
                        df[column] = encoder.fit_transform(df[column].astype(str))
                        self.encoders[column] = encoder
                    else:
                        if column in self.encoders:
                            df = self._safe_transform_encoder(df, column)

                elif unique_count <= 10:
                    # One-hot encoding for low cardinality
                    if fit:
                        dummies = pd.get_dummies(df[column], prefix=column, drop_first=True)
                        df = pd.concat([df.drop(column, axis=1), dummies], axis=1)
                        self.encoders[f"{column}_categories"] = df[column].unique()
                    else:
                        if f"{column}_categories" in self.encoders:
                            dummies = pd.get_dummies(df[column], prefix=column, drop_first=True)
                            df = pd.concat([df.drop(column, axis=1), dummies], axis=1)

                else:
                    # Target encoding for high cardinality (simplified version)
                    if fit and self.target_column in df.columns:
                        target_means = df.groupby(column)[self.target_column].mean()
                        df[f"{column}_target_encoded"] = df[column].map(target_means)
                        self.encoders[f"{column}_target_means"] = target_means
                    elif f"{column}_target_means" in self.encoders:
                        target_means = self.encoders[f"{column}_target_means"]
                        df[f"{column}_target_encoded"] = (
                            df[column].map(target_means).fillna(target_means.mean())
                        )

                    df = df.drop(column, axis=1)

        return df

    def scale_features(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """
        Advanced feature scaling with multiple methods

        Args:
            df: Input dataframe
            fit: Whether to fit scalers

        Returns:
            DataFrame with scaled features
        """
        df = df.copy()

        # Get numerical columns
        numerical_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        if self.target_column in numerical_columns:
            numerical_columns.remove(self.target_column)

        # Remove binary columns from scaling
        binary_columns = [col for col in numerical_columns if df[col].nunique() <= 2]
        numerical_columns = [col for col in numerical_columns if col not in binary_columns]

        if numerical_columns:
            if fit:
                # Choose and fit scaler
                if self.config.scaling_method == "standard":
                    scaler = StandardScaler()
                elif self.config.scaling_method == "power":
                    scaler = PowerTransformer(method="yeo-johnson", standardize=True)
                elif self.config.scaling_method == "minmax":
                    scaler = MinMaxScaler()
                elif self.config.scaling_method == "quantile":
                    scaler = QuantileTransformer(output_distribution="normal", random_state=42)
                else:  # robust (default)
                    scaler = RobustScaler()

                # Fit and transform
                df[numerical_columns] = scaler.fit_transform(df[numerical_columns])
                self.scalers["main"] = scaler
                self.scalers["columns"] = numerical_columns
            else:
                # Transform using existing scaler
                if "main" in self.scalers and "columns" in self.scalers:
                    # Only transform columns that exist in both datasets
                    cols_to_scale = [col for col in self.scalers["columns"] if col in df.columns]
                    if cols_to_scale:
                        df[cols_to_scale] = self.scalers["main"].transform(df[cols_to_scale])

        return df

    def select_features(
        self, df: pd.DataFrame, target: pd.Series, method: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Advanced feature selection using multiple methods

        Args:
            df: Input dataframe
            target: Target variable
            method: Selection method

        Returns:
            DataFrame with selected features
        """
        method = method or self.config.feature_selection_method

        # Exclude target from features
        feature_columns = df.columns.tolist()
        if self.target_column in feature_columns:
            feature_columns.remove(self.target_column)

        X = df[feature_columns]

        # Remove low variance features
        if self.config.variance_threshold > 0:
            variances = X.var()
            high_var_features = variances[variances > self.config.variance_threshold].index.tolist()
            X = X[high_var_features]
            feature_columns = high_var_features

        # Select features based on method
        k = min(self.config.n_features_to_select, len(feature_columns))

        if method == "mutual_info":
            selector = SelectKBest(score_func=mutual_info_classif, k=k)
        elif method == "chi2":
            # Ensure non-negative values for chi2
            X_positive = X - X.min() + 1e-8
            selector = SelectKBest(score_func=chi2, k=k)
            X_selected = selector.fit_transform(X_positive, target)
        else:  # f_classif
            selector = SelectKBest(score_func=f_classif, k=k)

        if method != "chi2":
            X_selected = selector.fit_transform(X, target)

        # Get selected features
        self.selected_features = X.columns[selector.get_support()].tolist()

        # Store feature importance scores
        feature_scores = pd.DataFrame(
            {"feature": X.columns, "score": selector.scores_}
        ).sort_values("score", ascending=False)

        self.feature_importance["univariate"] = feature_scores

        # Create dataframe with selected features
        df_selected = pd.DataFrame(X_selected, columns=self.selected_features, index=df.index)

        # Add target back if it exists
        if self.target_column in df.columns:
            df_selected[self.target_column] = df[self.target_column]

        return df_selected

    def apply_dimensionality_reduction(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """
        Apply PCA or other dimensionality reduction techniques

        Args:
            df: Input dataframe
            fit: Whether to fit the reducer

        Returns:
            DataFrame with reduced dimensions
        """
        if not self.config.enable_pca:
            return df

        df = df.copy()

        # Select numerical features
        feature_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        if self.target_column in feature_columns:
            feature_columns.remove(self.target_column)

        if len(feature_columns) > 10:  # Only apply if we have enough features
            X = df[feature_columns].fillna(0)

            if fit:
                # Use TruncatedSVD for sparse matrices or PCA
                if isinstance(self.config.pca_components, float):
                    # Find number of components for desired variance
                    pca_temp = PCA()
                    pca_temp.fit(X)
                    cumsum = np.cumsum(pca_temp.explained_variance_ratio_)
                    n_components = np.argmax(cumsum >= self.config.pca_components) + 1
                else:
                    n_components = min(self.config.pca_components, len(feature_columns))

                reducer = TruncatedSVD(n_components=n_components, random_state=42)
                X_reduced = reducer.fit_transform(X)

                self.dimensionality_reducers["pca"] = reducer
                self.dimensionality_reducers["n_components"] = n_components

                # Add PCA components as features
                for i in range(n_components):
                    df[f"pca_{i+1}"] = X_reduced[:, i]

                # Store explained variance
                self.feature_stats["pca_explained_variance"] = reducer.explained_variance_ratio_.sum()

            elif "pca" in self.dimensionality_reducers:
                reducer = self.dimensionality_reducers["pca"]
                X_reduced = reducer.transform(X)

                n_components = self.dimensionality_reducers["n_components"]
                for i in range(n_components):
                    df[f"pca_{i+1}"] = X_reduced[:, i]

        return df

    def _detect_outliers_iqr(self, series: pd.Series, multiplier: float = 1.5) -> np.ndarray:
        """
        Detect outliers using IQR method

        Args:
            series: Pandas series
            multiplier: IQR multiplier for outlier threshold

        Returns:
            Boolean array of outliers
        """
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - multiplier * IQR
        upper_bound = Q3 + multiplier * IQR
        return ((series < lower_bound) | (series > upper_bound)).astype(int)

    def _mahalanobis_distance(self, X: pd.DataFrame) -> np.ndarray:
        """
        Calculate Mahalanobis distance for anomaly detection

        Args:
            X: Feature matrix

        Returns:
            Array of Mahalanobis distances
        """
        mean = X.mean()
        cov = X.cov()
        cov_inv = np.linalg.pinv(cov)  # Pseudo-inverse for stability

        diff = X - mean
        md = np.sqrt(np.sum(diff.values @ cov_inv * diff.values, axis=1))

        return md

    def _safe_transform_encoder(self, df: pd.DataFrame, column: str) -> pd.DataFrame:
        """
        Safely transform categorical column handling unknown categories

        Args:
            df: Input dataframe
            column: Column to transform

        Returns:
            DataFrame with transformed column
        """
        encoder = self.encoders[column]

        # Handle unknown categories
        df[column] = df[column].astype(str)
        unknown_mask = ~df[column].isin(encoder.classes_)

        if unknown_mask.any():
            # Replace unknown with most frequent known category
            most_frequent = encoder.classes_[0]
            df.loc[unknown_mask, column] = most_frequent

        df[column] = encoder.transform(df[column])
        return df



    def fit_transform(self, df: pd.DataFrame, target_column: Optional[str] = None) -> pd.DataFrame:
        """
        Fit the complete feature engineering pipeline and transform data

        Args:
            df: Input dataframe
            target_column: Target column name

        Returns:
            Transformed dataframe
        """
        start_time = datetime.now()

        if target_column:
            self.target_column = target_column

        # Store original target
        target = df[self.target_column].copy() if self.target_column in df.columns else None

        print("🚀 Starting Advanced Feature Engineering Pipeline v2.0...")
        print(f"📊 Initial shape: {df.shape}")

        # Apply transformations in sequence
        pipeline_steps = [
            ("⏰ Time features", self.create_time_features),
            ("💰 Amount features", self.create_amount_features),
            ("🚄 Velocity features", self.create_velocity_features),
            ("📈 Statistical features", self.create_statistical_features),
            ("🔍 Anomaly features", lambda x: self.create_anomaly_features(x, fit=True)),
            ("🎯 Clustering features", lambda x: self.create_clustering_features(x, fit=True)),
            ("🔗 Interaction features", self.create_interaction_features),
            ("🔧 Missing value handling", self.handle_missing_values),
            ("🏷️ Categorical encoding", lambda x: self.encode_categorical_features(x, fit=True)),
            ("📏 Feature scaling", lambda x: self.scale_features(x, fit=True)),
        ]

        for step_name, step_func in pipeline_steps:
            print(f"\n{step_name}...")
            df = step_func(df)
            print(f"  ✅ Shape: {df.shape}")

        # Feature selection
        if target is not None and self.config.n_features_to_select > 0:
            print(f"\n🎯 Selecting top {self.config.n_features_to_select} features...")
            df = self.select_features(df, target)
            print(f"  ✅ Shape after selection: {df.shape}")

        # Dimensionality reduction
        if self.config.enable_pca:
            print("\n🗜️ Applying dimensionality reduction...")
            df = self.apply_dimensionality_reduction(df, fit=True)
            print(f"  ✅ Final shape: {df.shape}")

        self.is_fitted = True

        # Calculate pipeline execution time
        execution_time = (datetime.now() - start_time).total_seconds()

        print("\n✨ Feature Engineering Pipeline Completed!")
        print(f"⏱️ Total execution time: {execution_time:.2f} seconds")
        print(f"📊 Final dataset shape: {df.shape}")
        print(f"🎯 Features selected: {len(self.selected_features) if self.selected_features else 'All'}")

        # Save pipeline metadata
        self._save_pipeline_metadata(df, execution_time)

        return df

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform new data using fitted pipeline

        Args:
            df: Input dataframe

        Returns:
            Transformed dataframe
        """
        if not self.is_fitted:
            raise ValueError("Pipeline must be fitted before transforming new data!")

        # Apply transformations without fitting
        df = self.create_time_features(df)
        df = self.create_amount_features(df)
        df = self.create_velocity_features(df)
        df = self.create_statistical_features(df)
        df = self.create_anomaly_features(df, fit=False)
        df = self.create_clustering_features(df, fit=False)
        df = self.create_interaction_features(df)
        df = self.handle_missing_values(df)
        df = self.encode_categorical_features(df, fit=False)
        df = self.scale_features(df, fit=False)

        # Apply dimensionality reduction
        if self.config.enable_pca:
            df = self.apply_dimensionality_reduction(df, fit=False)

        # Select same features
        if self.selected_features:
            # NOTE: removed unused 'available_features' variable (flake8 F841)
            missing_features = [f for f in self.selected_features if f not in df.columns]

            if missing_features:
                # Add missing features as zeros
                for feature in missing_features:
                    df[feature] = 0

            df = df[self.selected_features]

            # Add target back if it exists
            if self.target_column in df.columns and self.target_column not in self.selected_features:
                df[self.target_column] = df[self.target_column]

        return df

    def _save_pipeline_metadata(self, df: pd.DataFrame, execution_time: float):
        """
        Save pipeline metadata for tracking and debugging

        Args:
            df: Transformed dataframe
            execution_time: Pipeline execution time
        """
        if not self.config.save_feature_metadata:
            return

        metadata = {
            "pipeline_version": self.pipeline_version,
            "execution_time": execution_time,
            "timestamp": datetime.now().isoformat(),
            "configuration": {
                "scaling_method": self.config.scaling_method,
                "feature_selection_method": self.config.feature_selection_method,
                "n_features_selected": len(self.selected_features) if self.selected_features else 0,
                "variance_threshold": self.config.variance_threshold,
                "pca_enabled": self.config.enable_pca,
                "pca_components": self.config.pca_components if self.config.enable_pca else None,
            },
            "feature_stats": {
                "total_features_created": df.shape[1],
                "numerical_features": len(df.select_dtypes(include=[np.number]).columns),
                "categorical_features": len(df.select_dtypes(include=["object"]).columns),
                "binary_features": len([col for col in df.columns if df[col].nunique() <= 2]),
            },
            "feature_creation_times": {k: v.isoformat() for k, v in self.feature_creation_time.items()},
            "selected_features": self.selected_features[:20] if self.selected_features else None,
            "feature_importance": {
                "top_10": self.feature_importance.get("univariate", pd.DataFrame())
                .head(10)
                .to_dict("records")
                if "univariate" in self.feature_importance
                else None
            },
        }

        # Save to file (in production, save to feature store or database)
        with open(
            f"feature_pipeline_metadata_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json", "w"
        ) as f:
            json.dump(metadata, f, indent=2)

    def get_feature_importance_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive feature importance report

        Returns:
            Dictionary containing feature importance information
        """
        report = {
            "pipeline_version": self.pipeline_version,
            "is_fitted": self.is_fitted,
            "total_features": len(self.selected_features) if self.selected_features else 0,
            "selected_features": self.selected_features,
            "feature_scores": self.feature_importance.get("univariate", pd.DataFrame()).to_dict("records")
            if "univariate" in self.feature_importance
            else None,
            "scaling_method": self.config.scaling_method,
            "selection_method": self.config.feature_selection_method,
            "pca_explained_variance": self.feature_stats.get("pca_explained_variance"),
            "encoders_fitted": list(self.encoders.keys()),
            "anomaly_detectors": list(self.anomaly_detectors.keys()),
            "clustering_models": list(self.clustering_models.keys()),
        }

        return report


# Testing and validation functions
def validate_pipeline_performance():
    """Validate the enhanced feature engineering pipeline"""
    import time
    import sys
    import os

    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    from data_processing.generate_data import create_fraud_dataset

    print("🧪 Testing Enhanced Feature Engineering Pipeline v2.0")
    print("=" * 60)

    # Create test dataset
    print("\n📊 Creating test dataset...")
    df_train = create_fraud_dataset(n_samples=5000)
    df_test = create_fraud_dataset(n_samples=1000)

    # Configure pipeline
    config = FeatureConfig(
        enable_velocity_features=True,
        enable_anomaly_features=True,
        enable_clustering_features=True,
        scaling_method="robust",
        feature_selection_method="mutual_info",
        n_features_to_select=50,
        enable_pca=True,
        pca_components=0.95,
    )

    # Initialize pipeline
    fe = AdvancedFeatureEngineering(config=config, target_column="Class")

    # Fit and transform training data
    print("\n🔧 Fitting pipeline on training data...")
    start_time = time.time()
    df_train_transformed = fe.fit_transform(df_train)
    fit_time = time.time() - start_time

    # Transform test data
    print("\n🔄 Transforming test data...")
    start_time = time.time()
    df_test_transformed = fe.transform(df_test)
    transform_time = time.time() - start_time

    # Generate report
    report = fe.get_feature_importance_report()

    # Print results
    print("\n" + "=" * 60)
    print("📊 PIPELINE PERFORMANCE SUMMARY")
    print("=" * 60)
    print(f"✅ Training samples: {len(df_train)}")
    print(f"✅ Test samples: {len(df_test)}")
    print(f"✅ Original features: {df_train.shape[1]}")
    print(f"✅ Features after engineering: {df_train_transformed.shape[1]}")
    print(f"✅ Selected features: {len(report['selected_features'])}")
    print(f"✅ Fit time: {fit_time:.2f} seconds")
    print(f"✅ Transform time: {transform_time:.2f} seconds")
    print(f"✅ PCA variance explained: {report.get('pca_explained_variance', 0):.2%}")

    print("\n🏆 Top 10 Features by Importance:")
    if report["feature_scores"]:
        for i, feature in enumerate(report["feature_scores"][:10], 1):
            print(f"  {i:2d}. {feature['feature']:30s} Score: {feature['score']:.4f}")

    print("\n✨ Pipeline validation completed successfully!")

    return df_train_transformed, df_test_transformed, fe


if __name__ == "__main__":
    # Run validation
    validate_pipeline_performance()
