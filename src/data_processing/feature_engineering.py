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

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, f_classif
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

class AdvancedFeatureEngineering:
    """
    Advanced feature engineering pipeline for fraud detection
    
    This class implements sophisticated feature engineering techniques
    commonly used in production fraud detection systems.
    """
    
    def __init__(self, target_column='Class'):
        """
        Initialize the feature engineering pipeline
        
        Args:
            target_column (str): Name of the target variable column
        """
        self.target_column = target_column
        self.scalers = {}
        self.encoders = {}
        self.feature_stats = {}
        self.selected_features = None
        self.is_fitted = False
        
    def create_time_features(self, df):
        """
        Extract comprehensive time-based features from timestamp
        
        Args:
            df (pd.DataFrame): Input dataframe with 'Time' column
            
        Returns:
            pd.DataFrame: Dataframe with additional time features
        """
        df = df.copy()
        
        # Convert time to datetime if needed
        if 'Time' in df.columns:
            # Assuming Time is in seconds from some reference point
            df['hour'] = (df['Time'] % (24 * 3600)) // 3600
            df['day_of_week'] = (df['Time'] // (24 * 3600)) % 7
            df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
            
            # Business hours (9 AM to 5 PM)
            df['is_business_hours'] = ((df['hour'] >= 9) & (df['hour'] <= 17)).astype(int)
            
            # Late night transactions (11 PM to 6 AM)
            df['is_late_night'] = ((df['hour'] >= 23) | (df['hour'] <= 6)).astype(int)
            
            # Peak hours (lunch: 12-14, evening: 18-20)
            df['is_peak_hours'] = (((df['hour'] >= 12) & (df['hour'] <= 14)) | 
                                 ((df['hour'] >= 18) & (df['hour'] <= 20))).astype(int)
        
        return df
    
    def create_amount_features(self, df):
        """
        Create sophisticated amount-based features
        
        Args:
            df (pd.DataFrame): Input dataframe with 'Amount' column
            
        Returns:
            pd.DataFrame: Dataframe with additional amount features
        """
        df = df.copy()
        
        if 'Amount' in df.columns:
            # Log transformation for skewed amounts
            df['amount_log'] = np.log1p(df['Amount'])
            
            # Amount categories
            df['amount_category'] = pd.cut(df['Amount'], 
                                         bins=[0, 10, 100, 1000, np.inf], 
                                         labels=['small', 'medium', 'large', 'very_large'])
            
            # Z-score normalization for outlier detection
            amount_mean = df['Amount'].mean()
            amount_std = df['Amount'].std()
            df['amount_zscore'] = (df['Amount'] - amount_mean) / amount_std
            df['is_amount_outlier'] = (np.abs(df['amount_zscore']) > 3).astype(int)
            
            # Round number indicator (psychological pricing)
            df['is_round_amount'] = (df['Amount'] % 1 == 0).astype(int)
            df['is_round_10'] = (df['Amount'] % 10 == 0).astype(int)
            df['is_round_100'] = (df['Amount'] % 100 == 0).astype(int)
        
        return df
    
    def create_statistical_features(self, df):
        """
        Create statistical features from the V1-V20 columns
        
        Args:
            df (pd.DataFrame): Input dataframe with V1-V20 columns
            
        Returns:
            pd.DataFrame: Dataframe with additional statistical features
        """
        df = df.copy()
        
        # Get V columns
        v_columns = [col for col in df.columns if col.startswith('V') and col[1:].isdigit()]
        
        if v_columns:
            v_data = df[v_columns]
            
            # Statistical aggregations
            df['v_mean'] = v_data.mean(axis=1)
            df['v_std'] = v_data.std(axis=1)
            df['v_skew'] = v_data.skew(axis=1)
            df['v_kurt'] = v_data.kurtosis(axis=1)
            df['v_median'] = v_data.median(axis=1)
            df['v_max'] = v_data.max(axis=1)
            df['v_min'] = v_data.min(axis=1)
            df['v_range'] = df['v_max'] - df['v_min']
            
            # Quartiles
            df['v_q25'] = v_data.quantile(0.25, axis=1)
            df['v_q75'] = v_data.quantile(0.75, axis=1)
            df['v_iqr'] = df['v_q75'] - df['v_q25']
            
            # Count of outliers in V features
            v_outliers = np.abs(stats.zscore(v_data, axis=1, nan_policy='omit')) > 2
            df['v_outlier_count'] = v_outliers.sum(axis=1)
            
            # Correlation with amount if available
            if 'Amount' in df.columns:
                for col in v_columns[:5]:  # First 5 V columns for computational efficiency
                    df[f'{col}_amount_ratio'] = df[col] / (df['Amount'] + 1e-8)
        
        return df
    
    def create_interaction_features(self, df):
        """
        Create interaction features between important variables
        
        Args:
            df (pd.DataFrame): Input dataframe
            
        Returns:
            pd.DataFrame: Dataframe with interaction features
        """
        df = df.copy()
        
        # Time-Amount interactions
        if 'hour' in df.columns and 'Amount' in df.columns:
            df['hour_amount_interaction'] = df['hour'] * df['amount_log']
            df['weekend_amount_interaction'] = df['is_weekend'] * df['amount_log']
            df['business_hours_amount'] = df['is_business_hours'] * df['amount_log']
        
        # V feature interactions (select most important ones)
        v_columns = [col for col in df.columns if col.startswith('V') and col[1:].isdigit()]
        if len(v_columns) >= 4:
            # Create interactions between highly correlated V features
            df['v1_v2_interaction'] = df['V1'] * df['V2']
            df['v3_v4_interaction'] = df['V3'] * df['V4']
            
            # Ratios of important V features
            df['v1_v3_ratio'] = df['V1'] / (np.abs(df['V3']) + 1e-8)
            df['v2_v4_ratio'] = df['V2'] / (np.abs(df['V4']) + 1e-8)
        
        return df
    
    def handle_missing_values(self, df):
        """
        Intelligent missing value handling strategy
        
        Args:
            df (pd.DataFrame): Input dataframe
            
        Returns:
            pd.DataFrame: Dataframe with handled missing values
        """
        df = df.copy()
        
        for column in df.columns:
            if df[column].isnull().any():
                if df[column].dtype == 'object':
                    # Categorical columns: fill with 'Unknown'
                    df[column] = df[column].fillna('Unknown')
                else:
                    # Numerical columns: fill with median (robust to outliers)
                    median_value = df[column].median()
                    df[column] = df[column].fillna(median_value)
                    
                    # Create missing indicator
                    df[f'{column}_was_missing'] = df[column].isnull().astype(int)
        
        return df
    
    def encode_categorical_features(self, df, fit=True):
        """
        Encode categorical features using appropriate encoding strategies
        
        Args:
            df (pd.DataFrame): Input dataframe
            fit (bool): Whether to fit encoders or use existing ones
            
        Returns:
            pd.DataFrame: Dataframe with encoded categorical features
        """
        df = df.copy()
        
        categorical_columns = df.select_dtypes(include=['object', 'category']).columns
        
        for column in categorical_columns:
            if column != self.target_column:
                if fit:
                    # Fit new encoder
                    encoder = LabelEncoder()
                    df[column] = encoder.fit_transform(df[column].astype(str))
                    self.encoders[column] = encoder
                else:
                    # Use existing encoder
                    if column in self.encoders:
                        # Handle unseen categories
                        known_categories = set(self.encoders[column].classes_)
                        df[column] = df[column].astype(str)
                        unknown_mask = ~df[column].isin(known_categories)
                        
                        if unknown_mask.any():
                            # Replace unknown categories with most frequent
                            most_frequent = self.encoders[column].classes_[0]
                            df.loc[unknown_mask, column] = most_frequent
                        
                        df[column] = self.encoders[column].transform(df[column])
        
        return df
    
    def scale_features(self, df, fit=True, method='robust'):
        """
        Scale numerical features using specified scaling method
        
        Args:
            df (pd.DataFrame): Input dataframe
            fit (bool): Whether to fit scalers or use existing ones
            method (str): Scaling method ('standard', 'robust')
            
        Returns:
            pd.DataFrame: Dataframe with scaled features
        """
        df = df.copy()
        
        # Get numerical columns (exclude target)
        numerical_columns = df.select_dtypes(include=[np.number]).columns
        if self.target_column in numerical_columns:
            numerical_columns = numerical_columns.drop(self.target_column)
        
        if fit:
            # Choose scaler
            if method == 'standard':
                scaler = StandardScaler()
            else:  # robust (default)
                scaler = RobustScaler()
            
            # Fit and transform
            df[numerical_columns] = scaler.fit_transform(df[numerical_columns])
            self.scalers['main'] = scaler
        else:
            # Transform using existing scaler
            if 'main' in self.scalers:
                df[numerical_columns] = self.scalers['main'].transform(df[numerical_columns])
        
        return df
    
    def select_features(self, df, target, k=50, method='f_classif'):
        """
        Select top k features using statistical tests
        
        Args:
            df (pd.DataFrame): Input dataframe
            target (pd.Series): Target variable
            k (int): Number of features to select
            method (str): Feature selection method
            
        Returns:
            pd.DataFrame: Dataframe with selected features
        """
        # Exclude target column from features
        feature_columns = df.columns.drop(self.target_column) if self.target_column in df.columns else df.columns
        X = df[feature_columns]
        
        # Select features
        if method == 'f_classif':
            selector = SelectKBest(score_func=f_classif, k=min(k, len(feature_columns)))
        else:
            selector = SelectKBest(score_func=f_classif, k=min(k, len(feature_columns)))
        
        X_selected = selector.fit_transform(X, target)
        
        # Get selected feature names
        self.selected_features = X.columns[selector.get_support()].tolist()
        
        # Create dataframe with selected features
        df_selected = pd.DataFrame(X_selected, columns=self.selected_features, index=df.index)
        
        # Add target back if it exists
        if self.target_column in df.columns:
            df_selected[self.target_column] = df[self.target_column]
        
        return df_selected
    
    def fit_transform(self, df, target_column=None):
        """
        Fit the feature engineering pipeline and transform the data
        
        Args:
            df (pd.DataFrame): Input dataframe
            target_column (str): Target column name (optional)
            
        Returns:
            pd.DataFrame: Transformed dataframe
        """
        if target_column:
            self.target_column = target_column
        
        # Store original target
        target = df[self.target_column] if self.target_column in df.columns else None
        
        print("🔧 Starting feature engineering pipeline...")
        
        # Apply transformations step by step
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
    
    def transform(self, df):
        """
        Transform new data using fitted pipeline
        
        Args:
            df (pd.DataFrame): Input dataframe
            
        Returns:
            pd.DataFrame: Transformed dataframe
        """
        if not self.is_fitted:
            raise ValueError("Pipeline must be fitted before transforming new data!")
        
        # Apply same transformations (without fitting)
        df = self.create_time_features(df)
        df = self.create_amount_features(df)
        df = self.create_statistical_features(df)
        df = self.create_interaction_features(df)
        df = self.handle_missing_values(df)
        df = self.encode_categorical_features(df, fit=False)
        df = self.scale_features(df, fit=False)
        
        # Select same features
        if self.selected_features:
            available_features = [f for f in self.selected_features if f in df.columns]
            df = df[available_features]
            
            # Add target if it exists
            if self.target_column in df.columns and self.target_column not in available_features:
                df[self.target_column] = df[self.target_column]
        
        return df
    
    def get_feature_importance_summary(self):
        """
        Get summary of feature engineering transformations
        
        Returns:
            dict: Summary of transformations applied
        """
        summary = {
            'total_features_created': len(self.selected_features) if self.selected_features else 0,
            'selected_features': self.selected_features,
            'encoders_fitted': list(self.encoders.keys()),
            'scalers_fitted': list(self.scalers.keys()),
            'is_fitted': self.is_fitted
        }
        
        return summary

# Example usage and testing function
def test_feature_engineering():
    """Test the feature engineering pipeline with sample data"""
    
    # Import data generation function
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    
    from data_processing.generate_data import create_fraud_dataset
    
    # Create sample data
    print("🔄 Creating sample dataset...")
    df = create_fraud_dataset(n_samples=10000)
    
    # Initialize feature engineering
    fe = AdvancedFeatureEngineering(target_column='Class')
    
    # Transform data
    df_transformed = fe.fit_transform(df)
    
    # Print results
    print(f"\n📊 Transformation Results:")
    print(f"Original shape: {df.shape}")
    print(f"Transformed shape: {df_transformed.shape}")
    print(f"Features created: {df_transformed.shape[1] - df.shape[1]}")
    
    # Print feature summary
    summary = fe.get_feature_importance_summary()
    print(f"\n🎯 Selected Features ({len(summary['selected_features'])}):")
    for i, feature in enumerate(summary['selected_features'][:10]):
        print(f"  {i+1}. {feature}")
    if len(summary['selected_features']) > 10:
        print(f"  ... and {len(summary['selected_features']) - 10} more")
    
    return df_transformed, fe

if __name__ == "__main__":
    # Run test
    test_feature_engineering()
