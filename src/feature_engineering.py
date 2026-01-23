"""
Feature Engineering Module

"""

import pandas as pd
from typing import Dict, List, Tuple

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from src.config import RANDOM_STATE


class FeatureEngineering:
    """
    Handles feature engineering and data preparation for modeling.
    """

    def __init__(self, store_data: pd.DataFrame):
        """
        Args:
            store_data: Cleaned, store-level DataFrame
        """
        self.store_data = store_data.copy()

    def add_interaction_features(self) -> pd.DataFrame:
        """
        Add only the required interaction feature:
        - avg_product_price × store_radius_hh_income
        """
        df = self.store_data.copy()

        if (
            'avg_product_price' in df.columns
            and 'store_radius_hh_income' in df.columns
        ):
        
            df['income_to_price_ratio'] = (
                df['store_radius_hh_income'] / df['avg_product_price'].clip(lower=1)
            )

        return df

    def get_feature_lists(
        self, df: pd.DataFrame
    ) -> Tuple[List[str], List[str]]:
        """
        Identify numerical and categorical features.
        """
        numerical_features = [
            'store_size',
            'store_radius_pop_density',
            'store_radius_hh_income',
            'avg_product_price',
            'income_to_price_ratio'
        ]

        categorical_features = [
            'storetype_id'
        ]

        # Keep only columns that actually exist
        numerical_features = [f for f in numerical_features if f in df.columns]
        categorical_features = [f for f in categorical_features if f in df.columns]

        return numerical_features, categorical_features

    def build_preprocessor(self, numerical_features: List[str], categorical_features: List[str]) -> ColumnTransformer:
        """
        Build preprocessing pipeline.
        """
        numeric_pipeline = Pipeline(
            steps=[
                ('scaler', StandardScaler())
            ]
        )

        categorical_pipeline = Pipeline(
            steps=[
                ('onehot', OneHotEncoder(handle_unknown='ignore', drop='first'))
            ]
        )

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_pipeline, numerical_features),
                ('cat', categorical_pipeline, categorical_features)
            ]
        )

        return preprocessor

    def prepare_data(self) -> Dict:
        """
        Full feature engineering workflow.
        """
        # 1. Add interaction feature
        df = self.add_interaction_features()

        # 2. Define target and features
        X = df.drop(
            columns=[
                'store_id',
                'city_id',
                'avg_monthly_revenue',
                'is_successful'
            ],
            errors='ignore'
        )
        y = df['is_successful']

        # 3. Train / Test split (STRATIFIED)
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            stratify=y,
            random_state=RANDOM_STATE
        )

        # 4. Identify feature types (from TRAIN only)
        numerical_features, categorical_features = self.get_feature_lists(X_train)

        # 5. Build preprocessing pipeline
        preprocessor = self.build_preprocessor(
            numerical_features,
            categorical_features
        )

        return {
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'preprocessor': preprocessor,
            'numerical_features': numerical_features,
            'categorical_features': categorical_features
        }


# Example usage
if __name__ == "__main__":
    from src.data_loader import DataLoader
    from src.data_cleaning import DataPreprocessor
    from src.eda import EDA

    loader = DataLoader()
    raw_data = loader.load_all_data()

    preprocessor = DataPreprocessor(raw_data)
    cleaned_data = preprocessor.preprocess_all()

    eda = EDA(
        cleaned_data['all_sales'],
        cleaned_data['store_cities']
    )
    store_data = eda.run_eda()

    fe = FeatureEngineering(store_data)
    data = fe.prepare_data()

    print("Feature engineering complete")
    print(f"Train size: {data['X_train'].shape}")
    print(f"Test size: {data['X_test'].shape}")
