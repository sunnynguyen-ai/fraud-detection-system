import os

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification


def create_fraud_dataset(n_samples=100000):
    """
    Create a realistic synthetic fraud detection dataset

    Returns:
        pd.DataFrame: Synthetic transaction data with fraud labels
    """
    np.random.seed(42)

    # Generate base features using make_classification
    X, y = make_classification(
        n_samples=n_samples,
        n_features=20,
        n_informative=15,
        n_redundant=3,
        n_clusters_per_class=1,
        weights=[0.99, 0.01],  # 1% fraud rate (realistic)
        random_state=42,
    )

    # Create realistic feature names
    feature_names = [
        "V1",
        "V2",
        "V3",
        "V4",
        "V5",
        "V6",
        "V7",
        "V8",
        "V9",
        "V10",
        "V11",
        "V12",
        "V13",
        "V14",
        "V15",
        "V16",
        "V17",
        "V18",
        "V19",
        "V20",
    ]

    # Create DataFrame
    df = pd.DataFrame(X, columns=feature_names)
    df["Class"] = y  # 0 = Normal, 1 = Fraud

    # Add realistic transaction amounts
    # Normal transactions: mostly small amounts
    # Fraud transactions: wider range, some very large
    normal_mask = df["Class"] == 0
    fraud_mask = df["Class"] == 1

    # Normal transaction amounts (log-normal distribution)
    df.loc[normal_mask, "Amount"] = np.random.lognormal(
        mean=3, sigma=1, size=normal_mask.sum()
    )

    # Fraud transaction amounts (more varied, some very large)
    fraud_amounts = []
    n_fraud = fraud_mask.sum()

    # 70% small fraud amounts
    fraud_amounts.extend(
        np.random.lognormal(mean=2.5, sigma=0.8, size=int(0.7 * n_fraud))
    )
    # 20% medium fraud amounts
    fraud_amounts.extend(
        np.random.lognormal(mean=4, sigma=0.5, size=int(0.2 * n_fraud))
    )
    # 10% large fraud amounts
    fraud_amounts.extend(
        np.random.lognormal(mean=6, sigma=0.3, size=n_fraud - len(fraud_amounts))
    )

    df.loc[fraud_mask, "Amount"] = fraud_amounts[:n_fraud]

    # Round amounts to 2 decimal places
    df["Amount"] = np.round(df["Amount"], 2)

    # Add time feature (seconds from some reference point)
    df["Time"] = np.random.randint(0, 172800, size=n_samples)  # 2 days worth of seconds

    # Sort by time to make it more realistic
    df = df.sort_values("Time").reset_index(drop=True)

    # Reorder columns to match credit card fraud dataset format
    columns_order = ["Time"] + feature_names + ["Amount", "Class"]
    df = df[columns_order]

    return df


def save_dataset(df, filepath="data/raw/fraud_data.csv"):
    """
    Save dataset to specified filepath

    Args:
        df (pd.DataFrame): Dataset to save
        filepath (str): Path where to save the dataset
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    # Save dataset
    df.to_csv(filepath, index=False)
    print(f"Dataset saved to {filepath}")


def generate_and_save_data():
    """Main function to generate and save fraud detection dataset"""
    print("Generating synthetic fraud detection dataset...")

    # Create dataset
    fraud_data = create_fraud_dataset(n_samples=100000)

    # Print dataset statistics
    print("\nDataset Statistics:")
    print(f"Total transactions: {len(fraud_data):,}")
    print(f"Fraud transactions: {fraud_data['Class'].sum():,}")
    print(f"Normal transactions: {(fraud_data['Class'] == 0).sum():,}")
    print(
        f"Fraud rate: {fraud_data['Class'].mean():.4f} "
        f"({fraud_data['Class'].mean() * 100:.2f}%)"
    )
    print(f"Average transaction amount: ${fraud_data['Amount'].mean():.2f}")
    print(f"Dataset shape: {fraud_data.shape}")

    # Save dataset
    save_dataset(fraud_data)

    # Display first few rows
    print("\nFirst 5 rows:")
    print(fraud_data.head())

    return fraud_data


if __name__ == "__main__":
    # Generate the dataset
    dataset = generate_and_save_data()
