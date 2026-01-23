"""
Q6 Exploratory Data Analysis (EDA)

This module performs EDA to understand store success patterns before modeling.
Focus: Features available for new store candidates (no promotions).

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple
from src.logger import logging
from src.exception import CustomException


class EDA:
    """
    Performs EDA for store success prediction.
    Only uses features that will be available for new store candidates.
    """
    
    def __init__(self, all_sales: pd.DataFrame, store_cities: pd.DataFrame):
        """Initialize EDA with cleaned datasets."""
        self.all_sales = all_sales
        self.store_cities = store_cities
        self.store_data = None
        
    def create_store_level_data(self) -> pd.DataFrame:
        """
        Create store-level data with features matching candidate store format.
        
        Features used (matching Q6c candidate columns):
        - store_size
        - storetype_id  
        - city_id
        - store_radius_pop_density
        - store_radius_hh_income
        - avg_product_price (if available)
        
        Target: avg_monthly_revenue
        """
        try:
            logging.info("Creating store-level data for modeling")
            
            # Step 1: Calculate average monthly revenue per store (target variable)
            store_monthly = self.all_sales.groupby(['store_id', 'year', 'month'])['revenue'].sum().reset_index()
            store_agg = store_monthly.groupby('store_id')['revenue'].mean().reset_index()
            store_agg.columns = ['store_id', 'avg_monthly_revenue']
            
            # Step 1b: Calculate average product price per store
            # This captures the pricing strategy/product mix of each store
            if 'price' in self.all_sales.columns:
                store_price = self.all_sales.groupby('store_id')['price'].median().reset_index()
                store_price.columns = ['store_id', 'avg_product_price']
                store_agg = store_agg.merge(store_price, on='store_id', how='left')
            
            # Step 2: Merge with store characteristics
            # These are the features we'll have for new candidate stores
            store_data = store_agg.merge(self.store_cities, on='store_id', how='inner')
            
            # Step 3: Define success flag (top 30% of stores)
            success_threshold = store_data['avg_monthly_revenue'].quantile(0.7)
            store_data['is_successful'] = (
                store_data['avg_monthly_revenue'] >= success_threshold
            ).astype(int)
            
            self.store_data = store_data
            
            logging.info(f"Created store-level data: {len(store_data)} stores")
            logging.info(f"Success threshold: ${success_threshold:,.2f}")
            logging.info(f"Successful stores: {store_data['is_successful'].sum()} ({store_data['is_successful'].mean()*100:.1f}%)")
            
            return store_data
            
        except Exception as e:
            raise CustomException(f"Error creating store-level data: {str(e)}")
    
    def analyze_data(self):
        """Analyze target and features available for candidate stores."""
        try:
            if self.store_data is None:
                self.create_store_level_data()
            
            # Target statistics
            target = self.store_data['avg_monthly_revenue']
            print(f"\nTarget Variable: Avg Monthly Revenue")
            print(f"  Mean:   ${target.mean():,.2f}")
            print(f"  Median: ${target.median():,.2f}")
            print(f"  Std:    ${target.std():,.2f}")
            print(f"  Min:    ${target.min():,.2f}")
            print(f"  Max:    ${target.max():,.2f}")
            
            # Store type analysis
            
            print(f"Store Type Counts:")
            print(self.store_data['storetype_id'].value_counts())
            
            print(f"\nAvg Revenue by Store Type:")
            type_revenue = self.store_data.groupby('storetype_id')['avg_monthly_revenue'].mean().sort_values(ascending=False)
            for store_type, revenue in type_revenue.items():
                print(f"  {store_type}: ${revenue:,.2f}")
            
            # Identify modeling features (matching candidate store columns)
            model_features = [
                'store_size',
                'storetype_id',
                'city_id', 
                'store_radius_pop_density',
                'store_radius_hh_income',
                'avg_product_price'  # Calculated from sales data
            ]
            
            # Filter to available features
            available_features = [f for f in model_features if f in self.store_data.columns]
            
            print(f"\n--- Modeling Features (matching candidate store format) ---")
            for feat in available_features:
                if self.store_data[feat].dtype in ['object', 'category']:
                    n_unique = self.store_data[feat].nunique()
                    print(f"  {feat} (categorical, {n_unique} unique values)")
                else:
                    print(f"  {feat} (numeric)")
            
            # Numeric features for correlation
            numeric_features = [
                f for f in available_features 
                if self.store_data[f].dtype in ['int64', 'float64']
            ]
            
            if numeric_features:
                # Correlations with target
                correlations = self.store_data[numeric_features + ['avg_monthly_revenue']].corr()['avg_monthly_revenue'].sort_values(ascending=False)
                
                print(f"\n--- Correlations with Revenue ---")
                for feat, corr in correlations.items():
                    if feat != 'avg_monthly_revenue':
                        strength = "Strong" if abs(corr) > 0.5 else "Moderate" if abs(corr) > 0.3 else "Weak"
                        print(f"  {feat:35s}: {corr:6.3f} ({strength})")
            
            # Missing values
            missing = self.store_data[available_features].isnull().sum()
            if missing.sum() > 0:
                print(f"\n Missing Values Found:")
                print(missing[missing > 0])
            else:
                print(f"\n No missing values in modeling features")
            
            return available_features
            
        except Exception as e:
            raise CustomException(f"Error analyzing data: {str(e)}")
    
    def create_visualisations(self, save_path: str = None):
        """Create key EDA visualisations for modeling features."""
        try:
            if self.store_data is None:
                self.create_store_level_data()
            
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            
            # Plot 1: Target distribution
            axes[0, 0].hist(self.store_data['avg_monthly_revenue'], bins=30, edgecolor='black')
            axes[0, 0].axvline(self.store_data['avg_monthly_revenue'].median(), 
                              color='r', linestyle='--', label='Median')
            axes[0, 0].set_title('Distribution of Avg Monthly Revenue')
            axes[0, 0].set_xlabel('Avg Monthly Revenue ($)')
            axes[0, 0].set_ylabel('Frequency')
            axes[0, 0].legend()
            
            # Plot 2: Store size vs revenue
            axes[0, 1].scatter(self.store_data['store_size'], 
                             self.store_data['avg_monthly_revenue'], alpha=0.6)
            axes[0, 1].set_title('Store Size vs Revenue')
            axes[0, 1].set_xlabel('Store Size')
            axes[0, 1].set_ylabel('Avg Monthly Revenue ($)')
            
            # Plot 3: Store radius population density vs revenue
            axes[0, 2].scatter(self.store_data['store_radius_pop_density'], 
                             self.store_data['avg_monthly_revenue'], alpha=0.6, color='green')
            axes[0, 2].set_title('Population Density vs Revenue')
            axes[0, 2].set_xlabel('Store Radius Pop Density')
            axes[0, 2].set_ylabel('Avg Monthly Revenue ($)')
            
            # Plot 4: Store radius household income vs revenue
            axes[1, 0].scatter(self.store_data['store_radius_hh_income'], 
                             self.store_data['avg_monthly_revenue'], alpha=0.6, color='orange')
            axes[1, 0].set_title('Household Income vs Revenue')
            axes[1, 0].set_xlabel('Store Radius HH Income ($)')
            axes[1, 0].set_ylabel('Avg Monthly Revenue ($)')
            
            # Plot 5: Revenue by store type (boxplot)
            if 'storetype_id' in self.store_data.columns:
                self.store_data.boxplot(column='avg_monthly_revenue', by='storetype_id', ax=axes[1, 1])
                axes[1, 1].set_title('Revenue by Store Type')
                axes[1, 1].set_xlabel('Store Type')
                axes[1, 1].set_ylabel('Avg Monthly Revenue ($)')
                plt.suptitle('')  # Remove default title
            
            # Plot 6: Success rate by store type
            if 'storetype_id' in self.store_data.columns:
                success_by_type = self.store_data.groupby('storetype_id')['is_successful'].mean()
                success_by_type.plot(kind='bar', ax=axes[1, 2], color='steelblue')
                axes[1, 2].set_title('Success Rate by Store Type')
                axes[1, 2].set_xlabel('Store Type')
                axes[1, 2].set_ylabel('Success Rate')
                axes[1, 2].tick_params(axis='x', rotation=45)
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logging.info(f"EDA plots saved to {save_path}")
            
            plt.show()
            
        except Exception as e:
            raise CustomException(f"Error creating visualisations: {str(e)}")
    
    def run_eda(self, save_plot: bool = False) -> pd.DataFrame:
        """
        Run complete EDA pipeline.
        
        Returns:
            pd.DataFrame: Store-level data ready for modeling with only
                         features available for candidate stores
        """
        try:
            
            logging.info("Starting Q6 EDA: Store Success Analysis")
            
            
            # Create store-level data
            store_data = self.create_store_level_data()
            
            # Analyze features
            available_features = self.analyze_data()
            
            # Visualize
            plot_path = 'plots/q6_eda.png' if save_plot else None
            self.create_visualisations(save_path=plot_path)
            
            
            print(f"Dataset: {len(store_data)} stores")
            print(f"Target: avg_monthly_revenue")
            print(f"Features: {len(available_features)} (matching candidate store format)")
            
            return store_data
            
        except Exception as e:
            raise CustomException(f"Error in EDA: {str(e)}")


# Example usage
if __name__ == "__main__":
    from src.data_loader import DataLoader
    from src.data_cleaning import DataPreprocessor
    
    # Load and preprocess
    loader = DataLoader()
    raw_data = loader.load_all_data()
    
    preprocessor = DataPreprocessor(raw_data)
    cleaned_data = preprocessor.preprocess_all()
    
    # Run EDA
    eda = EDA(cleaned_data['all_sales'], cleaned_data['store_cities'])
    store_data = eda.run_eda(save_plot=True)
    
    # Save for modeling
    store_data.to_csv('data/processed/q6_store_data.csv', index=False)
    logging.info("Store data saved for modeling")