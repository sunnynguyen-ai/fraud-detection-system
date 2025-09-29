"""
Unit tests for feature engineering pipeline
"""

import os
import sys

import numpy as np
import pytest

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_processing.feature_engineering import AdvancedFeatureEngineering
from src.data_processing.generate_data import create_fraud_dataset


class TestFeatureEngineering:
    """Test cases for feature engineering"""

    @pytest.fixture
    def sample_dataframe(self):
        """Create sample dataframe for testing"""
        return create_fraud_dataset(n_samples=100)

    def test_time_features(self, sample_dataframe):
        """Test time feature creation"""
        fe = AdvancedFeatureEngineering()
        df_with_time = fe.create_time_features(sample_dataframe)

        # Check new columns exist
        assert "hour" in df_with_time.columns
        assert "day_of_week" in df_with_time.columns
        assert "is_weekend" in df_with_time.columns
        assert "is_business_hours" in df_with_time.columns

        # Check value ranges
        assert df_with_time["hour"].min() >= 0
        assert df_with_time["hour"].max() < 24
        assert df_with_time["day_of_week"].min() >= 0
        assert df_with_time["day_of_week"].max() < 7

    def test_amount_features(self, sample_dataframe):
        """Test amount feature creation"""
        fe = AdvancedFeatureEngineering()
        df_with_amount = fe.create_amount_features(sample_dataframe)

        # Check new columns exist
        assert "amount_log" in df_with_amount.columns
        assert "amount_category" in df_with_amount.columns
        assert "amount_zscore" in df_with_amount.columns
        assert "is_round_amount" in df_with_amount.columns

        # Check transformations
        assert np.all(df_with_amount["amount_log"] >= 0)
        assert df_with_amount["is_round_amount"].isin([0, 1]).all()

    def test_statistical_features(self, sample_dataframe):
        """Test statistical feature creation"""
        fe = AdvancedFeatureEngineering()
        df_with_stats = fe.create_statistical_features(sample_dataframe)

        # Check new columns exist
        assert "v_mean" in df_with_stats.columns
        assert "v_std" in df_with_stats.columns
        assert "v_max" in df_with_stats.columns
        assert "v_min" in df_with_stats.columns
        assert "v_outlier_count" in df_with_stats.columns

    def test_fit_transform_pipeline(self, sample_dataframe):
        """Test complete fit_transform pipeline"""
        fe = AdvancedFeatureEngineering(target_column="Class")

        # Fit and transform
        df_transformed = fe.fit_transform(sample_dataframe)

        assert fe.is_fitted
        assert len(df_transformed) == len(sample_dataframe)
        assert df_transformed.shape[1] >= sample_dataframe.shape[1]

        # Check no missing values
        assert not df_transformed.isnull().any().any()

    def test_transform_new_data(self, sample_dataframe):
        """Test transforming new data with fitted pipeline"""
        fe = AdvancedFeatureEngineering(target_column="Class")

        # Fit on first half
        train_df = sample_dataframe.iloc[:50]
        test_df = sample_dataframe.iloc[50:]

        fe.fit_transform(train_df)
        assert fe.is_fitted

        # Transform second half
        transformed_test = fe.transform(test_df)
        assert len(transformed_test) == len(test_df)

        # Check columns match
        train_transformed = fe.transform(train_df)
        assert list(transformed_test.columns) == list(train_transformed.columns)
