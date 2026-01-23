import pandas as pd
from typing import Dict
from src.logger import logging
from src.exception import CustomException



class DataPreprocessor:
    """
    Handles data cleaning and preprocessing for Pawsible Apparel datasets.

    Attributes:
        all_sales (pd.DataFrame)
        product_hierarchy (pd.DataFrame)
        store_cities (pd.DataFrame)
    """

    def __init__(self, data_dict: Dict[str, pd.DataFrame]):
        """
        Initialize preprocessor with raw data.
        """
        self.all_sales = data_dict['all_sales'].copy()
        self.product_hierarchy = data_dict['product_hierarchy'].copy()
        self.store_cities = data_dict['store_cities'].copy()

    def clean_all_sales(self) -> pd.DataFrame:
        """Clean all_sales dataset."""
        try:
            df = self.all_sales.copy()
            initial_rows = len(df)
            logging.info(f"Starting all_sales cleaning with {initial_rows} rows")

            # Remove duplicates
            df = df.drop_duplicates()
            logging.info(f"Removed {initial_rows - len(df)} duplicate rows")

            # Ensure date column is datetime
            if not pd.api.types.is_datetime64_any_dtype(df['date']):
                df['date'] = pd.to_datetime(df['date'], errors='coerce')

            # Drop rows with invalid dates
            df = df.dropna(subset=['date'])

            # Extract date features
            df['year'] = df['date'].dt.year
            df['month'] = df['date'].dt.month
            df['day'] = df['date'].dt.day

            rev_cols = ['revenue']
            for col in rev_cols:
                if col in df.columns:
                    if df[col].dtype == 'object':
                        df[col] = df[col].str.replace('$', '', regex=False).str.replace(',', '', regex=False)
                    df[col] = pd.to_numeric(df[col], errors='coerce')

            # Handle missing numeric values
            numeric_cols = ['sales', 'revenue', 'stock', 'price']
            for col in numeric_cols:
                if col in df.columns:
                    if col in ['sales', 'stock']:
                        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
                    else:
                        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(df[col].median())

            # Remove negative values for key numeric columns
            for col in numeric_cols:
                df = df[df[col] >= 0]

            # Handle promotion columns (fill missing strings with 'None')
            promo_cols = [col for col in df.columns if 'promo' in col.lower()]
            for col in promo_cols:
                if df[col].dtype == 'object':
                    df[col] = df[col].fillna('None')

            logging.info(f"Cleaned all_sales: {len(df)} rows remaining")
            return df

        except Exception as e:
            raise CustomException(f"Error cleaning all_sales: {str(e)}")

    def clean_product_hierarchy(self) -> pd.DataFrame:
        """Clean product_hierarchy dataset."""
        try:
            df = self.product_hierarchy.copy()
            initial_rows = len(df)
            logging.info(f"Starting product_hierarchy cleaning with {initial_rows} rows")

            # Remove duplicate product_ids
            df = df.drop_duplicates(subset=['product_id'], keep='first')
            logging.info(f"Removed {initial_rows - len(df)} duplicate product_ids")

            # Drop rows with missing dimensions
            dimension_cols = ['product_length', 'product_depth', 'product_width']
            df = df.dropna(subset=dimension_cols)

            # Ensure numeric and positive dimensions
            for col in dimension_cols:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                df = df[df[col] > 0]

            # Calculate product volume
            df['product_volume'] = df['product_length'] * df['product_depth'] * df['product_width']

            logging.info(f"Cleaned product_hierarchy: {len(df)} products remaining")
            return df

        except Exception as e:
            raise CustomException(f"Error cleaning product_hierarchy: {str(e)}")

    def clean_store_cities(self) -> pd.DataFrame:
        """Clean store_cities dataset."""
        try:
            df = self.store_cities.copy()
            initial_rows = len(df)
            logging.info(f"Starting store_cities cleaning with {initial_rows} rows")

            # Remove duplicate store_ids
            df = df.drop_duplicates(subset=['store_id'], keep='first')
            logging.info(f"Removed {initial_rows - len(df)} duplicate store_ids")

            # Clean income columns (remove $ and commas, convert to numeric)
            income_cols = ['city_hh_income', 'store_radius_hh_income']
            for col in income_cols:
                if col in df.columns:
                    if df[col].dtype == 'object':
                        df[col] = df[col].str.replace('$', '', regex=False).str.replace(',', '', regex=False)
                    df[col] = pd.to_numeric(df[col], errors='coerce')

            # Fill missing numeric values with median
            numeric_cols = ['store_size', 'city_pop_density', 'city_hh_income',
                            'store_radius_pop_density', 'store_radius_hh_income', 'avg_product_price']
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                    df[col] = df[col].fillna(df[col].median())

            # Ensure positive values for store_size
            if 'store_size' in df.columns:
                df = df[df['store_size'] > 0]

            logging.info(f"Cleaned store_cities: {len(df)} stores remaining")
            return df

        except Exception as e:
            raise CustomException(f"Error cleaning store_cities: {str(e)}")

    def preprocess_all(self) -> Dict[str, pd.DataFrame]:
        """
        Run full preprocessing pipeline for all datasets.

        Returns:
            dict: keys = 'all_sales', 'product_hierarchy', 'store_cities'
        """
        try:
            logging.info("Starting full preprocessing pipeline")

            all_sales_clean = self.clean_all_sales()
            product_hierarchy_clean = self.clean_product_hierarchy()
            store_cities_clean = self.clean_store_cities()

            logging.info("Preprocessing pipeline completed successfully")


            return {
                "all_sales": all_sales_clean,
                "product_hierarchy": product_hierarchy_clean,
                "store_cities": store_cities_clean
            }

        except Exception as e:
            raise CustomException(f"Error in preprocessing pipeline: {str(e)}")

   

# Example usage
if __name__ == "__main__":
    from src.data_loader import DataLoader

    loader = DataLoader()
    raw_data = loader.load_all_data()

    preprocessor = DataPreprocessor(raw_data)
    cleaned_data = preprocessor.preprocess_all()

    logging.info(f"Final shapes - Sales: {cleaned_data['all_sales'].shape}, "
                 f"Products: {cleaned_data['product_hierarchy'].shape}, "
                 f"Stores: {cleaned_data['store_cities'].shape}")
