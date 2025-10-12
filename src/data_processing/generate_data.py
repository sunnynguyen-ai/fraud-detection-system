"""
Synthetic Fraud Detection Dataset Generator

This module creates realistic synthetic fraud detection datasets for machine learning
model development and testing. It simulates credit card transaction patterns with
configurable fraud rates and realistic feature distributions.
"""

import os
import logging
from typing import Tuple, Optional
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class DatasetConfig:
    """Configuration for synthetic fraud dataset generation."""
    n_samples: int = 100000
    fraud_rate: float = 0.01
    n_features: int = 20
    n_informative: int = 15
    n_redundant: int = 3
    random_state: int = 42
    time_period_days: int = 2


class FraudDatasetGenerator:
    """Generate synthetic fraud detection datasets with realistic patterns."""
    
    def __init__(self, config: Optional[DatasetConfig] = None):
        """
        Initialize the generator with configuration.
        
        Args:
            config: DatasetConfig object with generation parameters
        """
        self.config = config or DatasetConfig()
        np.random.seed(self.config.random_state)
        
    def create_fraud_dataset(self) -> pd.DataFrame:
        """
        Create a realistic synthetic fraud detection dataset.
        
        Returns:
            pd.DataFrame: Synthetic transaction data with fraud labels
        """
        logger.info(f"Generating {self.config.n_samples:,} synthetic transactions...")
        
        # Generate base features
        X, y = self._generate_base_features()
        
        # Create DataFrame with PCA-transformed features (mimicking real credit card data)
        feature_names = [f"V{i}" for i in range(1, self.config.n_features + 1)]
        df = pd.DataFrame(X, columns=feature_names)
        df["Class"] = y
        
        # Add realistic transaction amounts
        df["Amount"] = self._generate_transaction_amounts(df["Class"])
        
        # Add temporal features
        df = self._add_temporal_features(df)
        
        # Add merchant and customer features
        df = self._add_categorical_features(df)
        
        # Reorder columns
        base_cols = ["Time", "Amount", "Class"]
        feature_cols = [col for col in df.columns if col not in base_cols]
        df = df[["Time"] + feature_cols + ["Amount", "Class"]]
        
        # Sort by time for realism
        df = df.sort_values("Time").reset_index(drop=True)
        
        return df
    
    def _generate_base_features(self) -> Tuple[np.ndarray, np.ndarray]:
        """Generate base features using sklearn's make_classification."""
        return make_classification(
            n_samples=self.config.n_samples,
            n_features=self.config.n_features,
            n_informative=self.config.n_informative,
            n_redundant=self.config.n_redundant,
            n_clusters_per_class=1,
            weights=[1 - self.config.fraud_rate, self.config.fraud_rate],
            random_state=self.config.random_state,
            flip_y=0.01,  # Add some label noise for realism
        )
    
    def _generate_transaction_amounts(self, labels: pd.Series) -> np.ndarray:
        """
        Generate realistic transaction amounts based on fraud labels.
        
        Args:
            labels: Series of fraud labels (0=normal, 1=fraud)
            
        Returns:
            Array of transaction amounts
        """
        amounts = np.zeros(len(labels))
        
        # Normal transactions: log-normal distribution with realistic parameters
        normal_mask = labels == 0
        if normal_mask.sum() > 0:
            # Mix of small and medium transactions
            n_normal = normal_mask.sum()
            small_txns = np.random.lognormal(mean=2.5, sigma=1.2, size=int(0.8 * n_normal))
            medium_txns = np.random.lognormal(mean=4, sigma=0.8, size=n_normal - len(small_txns))
            normal_amounts = np.concatenate([small_txns, medium_txns])
            np.random.shuffle(normal_amounts)
            amounts[normal_mask] = normal_amounts
        
        # Fraud transactions: more varied distribution
        fraud_mask = labels == 1
        if fraud_mask.sum() > 0:
            n_fraud = fraud_mask.sum()
            fraud_amounts = []
            
            # 60% small fraud (testing cards)
            fraud_amounts.extend(
                np.random.lognormal(mean=2.0, sigma=0.8, size=int(0.6 * n_fraud))
            )
            # 25% medium fraud
            fraud_amounts.extend(
                np.random.lognormal(mean=4.5, sigma=0.7, size=int(0.25 * n_fraud))
            )
            # 15% large fraud (big purchases)
            remaining = n_fraud - len(fraud_amounts)
            fraud_amounts.extend(
                np.random.lognormal(mean=6.5, sigma=0.5, size=remaining)
            )
            
            amounts[fraud_mask] = fraud_amounts[:n_fraud]
        
        # Round to 2 decimal places and cap at reasonable maximum
        amounts = np.round(amounts, 2)
        amounts = np.clip(amounts, 0.01, 25000)  # Max transaction of $25,000
        
        return amounts
    
    def _add_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add temporal features to the dataset.
        
        Args:
            df: DataFrame with transaction data
            
        Returns:
            DataFrame with added temporal features
        """
        # Base time in seconds
        total_seconds = self.config.time_period_days * 24 * 3600
        df["Time"] = np.random.randint(0, total_seconds, size=len(df))
        
        # Add hour of day feature (useful for fraud detection)
        df["Hour"] = (df["Time"] // 3600) % 24
        
        # Add day of week (0-6)
        df["DayOfWeek"] = (df["Time"] // 86400) % 7
        
        # Fraud is more likely during certain hours (night time and early morning)
        fraud_mask = df["Class"] == 1
        if fraud_mask.sum() > 0:
            # Bias fraud towards night hours (0-6) and late evening (22-24)
            fraud_hours = df.loc[fraud_mask, "Hour"].values
            night_mask = np.random.random(len(fraud_hours)) < 0.4
            fraud_hours[night_mask] = np.random.choice([0, 1, 2, 3, 4, 5, 22, 23], 
                                                       size=night_mask.sum())
            df.loc[fraud_mask, "Hour"] = fraud_hours
            
            # Recalculate Time based on modified Hour
            df.loc[fraud_mask, "Time"] = (
                df.loc[fraud_mask, "DayOfWeek"] * 86400 + 
                df.loc[fraud_mask, "Hour"] * 3600 + 
                np.random.randint(0, 3600, size=fraud_mask.sum())
            )
        
        return df
    
    def _add_categorical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add categorical features like merchant category and customer segment.
        
        Args:
            df: DataFrame with transaction data
            
        Returns:
            DataFrame with added categorical features
        """
        # Merchant Category Code (MCC) - simplified version
        mcc_categories = {
            0: "Grocery",
            1: "Gas Station", 
            2: "Restaurant",
            3: "Online Shopping",
            4: "Entertainment",
            5: "Travel",
            6: "Cash Advance",
            7: "Other"
        }
        
        # Normal transactions distribution
        normal_mcc_probs = [0.25, 0.15, 0.20, 0.15, 0.10, 0.05, 0.02, 0.08]
        # Fraud transactions have different patterns
        fraud_mcc_probs = [0.10, 0.25, 0.05, 0.30, 0.05, 0.10, 0.10, 0.05]
        
        df["MerchantCategory"] = 0
        
        normal_mask = df["Class"] == 0
        fraud_mask = df["Class"] == 1
        
        if normal_mask.sum() > 0:
            df.loc[normal_mask, "MerchantCategory"] = np.random.choice(
                list(mcc_categories.keys()), 
                size=normal_mask.sum(),
                p=normal_mcc_probs
            )
        
        if fraud_mask.sum() > 0:
            df.loc[fraud_mask, "MerchantCategory"] = np.random.choice(
                list(mcc_categories.keys()),
                size=fraud_mask.sum(),
                p=fraud_mcc_probs
            )
        
        # Customer risk score (derived feature)
        df["CustomerRiskScore"] = np.random.beta(2, 5, size=len(df))
        # Fraud more likely with higher risk scores
        df.loc[fraud_mask, "CustomerRiskScore"] = np.random.beta(5, 2, size=fraud_mask.sum())
        df["CustomerRiskScore"] = np.round(df["CustomerRiskScore"], 3)
        
        return df
    
    def get_dataset_statistics(self, df: pd.DataFrame) -> dict:
        """
        Calculate and return dataset statistics.
        
        Args:
            df: DataFrame with transaction data
            
        Returns:
            Dictionary with dataset statistics
        """
        stats = {
            "total_transactions": len(df),
            "fraud_transactions": df["Class"].sum(),
            "normal_transactions": (df["Class"] == 0).sum(),
            "fraud_rate": df["Class"].mean(),
            "avg_transaction_amount": df["Amount"].mean(),
            "median_transaction_amount": df["Amount"].median(),
            "max_transaction_amount": df["Amount"].max(),
            "min_transaction_amount": df["Amount"].min(),
            "dataset_shape": df.shape,
            "memory_usage_mb": df.memory_usage(deep=True).sum() / 1024**2
        }
        
        # Add fraud vs normal amount statistics
        stats["avg_normal_amount"] = df[df["Class"] == 0]["Amount"].mean()
        stats["avg_fraud_amount"] = df[df["Class"] == 1]["Amount"].mean()
        
        return stats
    
    def save_dataset(self, df: pd.DataFrame, filepath: str = "data/raw/fraud_data.csv",
                    include_metadata: bool = True) -> None:
        """
        Save dataset to specified filepath with optional metadata.
        
        Args:
            df: Dataset to save
            filepath: Path where to save the dataset
            include_metadata: Whether to save metadata file
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Save dataset
        df.to_csv(filepath, index=False)
        logger.info(f"Dataset saved to {filepath}")
        
        # Save metadata if requested
        if include_metadata:
            metadata_path = filepath.replace('.csv', '_metadata.json')
            stats = self.get_dataset_statistics(df)
            
            import json
            with open(metadata_path, 'w') as f:
                json.dump(stats, f, indent=2, default=str)
            logger.info(f"Metadata saved to {metadata_path}")
    
    def validate_dataset(self, df: pd.DataFrame) -> bool:
        """
        Validate the generated dataset for consistency.
        
        Args:
            df: DataFrame to validate
            
        Returns:
            True if dataset is valid, False otherwise
        """
        checks = []
        
        # Check for missing values
        checks.append(df.isnull().sum().sum() == 0)
        
        # Check fraud rate is within expected range
        actual_fraud_rate = df["Class"].mean()
        expected_fraud_rate = self.config.fraud_rate
        checks.append(abs(actual_fraud_rate - expected_fraud_rate) < 0.005)
        
        # Check all amounts are positive
        checks.append((df["Amount"] > 0).all())
        
        # Check time values are within range
        max_time = self.config.time_period_days * 24 * 3600
        checks.append((df["Time"] >= 0).all() and (df["Time"] < max_time).all())
        
        # Check class labels are binary
        checks.append(set(df["Class"].unique()) <= {0, 1})
        
        if not all(checks):
            logger.warning("Dataset validation failed!")
            return False
        
        logger.info("Dataset validation passed ✓")
        return True


def generate_and_save_data(config: Optional[DatasetConfig] = None) -> pd.DataFrame:
    """
    Main function to generate and save fraud detection dataset.
    
    Args:
        config: Optional configuration for dataset generation
        
    Returns:
        Generated DataFrame
    """
    # Initialize generator
    generator = FraudDatasetGenerator(config)
    
    # Create dataset
    fraud_data = generator.create_fraud_dataset()
    
    # Validate dataset
    if not generator.validate_dataset(fraud_data):
        raise ValueError("Generated dataset failed validation")
    
    # Get and display statistics
    stats = generator.get_dataset_statistics(fraud_data)
    
    print("\n" + "="*50)
    print("DATASET STATISTICS")
    print("="*50)
    print(f"Total transactions: {stats['total_transactions']:,}")
    print(f"Fraud transactions: {stats['fraud_transactions']:,}")
    print(f"Normal transactions: {stats['normal_transactions']:,}")
    print(f"Fraud rate: {stats['fraud_rate']:.4f} ({stats['fraud_rate']*100:.2f}%)")
    print(f"\nTransaction Amounts:")
    print(f"  Average: ${stats['avg_transaction_amount']:.2f}")
    print(f"  Median: ${stats['median_transaction_amount']:.2f}")
    print(f"  Range: ${stats['min_transaction_amount']:.2f} - ${stats['max_transaction_amount']:.2f}")
    print(f"  Normal avg: ${stats['avg_normal_amount']:.2f}")
    print(f"  Fraud avg: ${stats['avg_fraud_amount']:.2f}")
    print(f"\nDataset shape: {stats['dataset_shape']}")
    print(f"Memory usage: {stats['memory_usage_mb']:.2f} MB")
    
    # Save dataset
    generator.save_dataset(fraud_data)
    
    # Display sample
    print("\n" + "="*50)
    print("SAMPLE DATA (First 5 rows)")
    print("="*50)
    print(fraud_data.head())
    
    print("\n" + "="*50)
    print("FEATURE INFORMATION")
    print("="*50)
    print(fraud_data.dtypes)
    
    return fraud_data


if __name__ == "__main__":
    # Example: Generate with custom configuration
    custom_config = DatasetConfig(
        n_samples=100000,
        fraud_rate=0.01,
        time_period_days=7,  # One week of data
        random_state=42
    )
    
    # Generate the dataset
    dataset = generate_and_save_data(custom_config)
