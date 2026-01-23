import os
import pandas as pd
from src.config import ALL_SALES_CSV, PRODUCT_HIERARCHY_CSV, STORE_CITIES_CSV
from src.exception import CustomException
from src.logger import logging

class DataLoader:
    """
    DataLoader class for loading and validating raw CSV data.
    """

    def __init__(self):
        self.all_sales_path = ALL_SALES_CSV
        self.product_hierarchy_path = PRODUCT_HIERARCHY_CSV
        self.store_cities_path = STORE_CITIES_CSV

    def _check_file_exists(self, path: str):
        """
        Checks if a file exists at the given path.
        Raises CustomException if not found.
        """
        if not os.path.exists(path):
            raise CustomException(f"File not found: {path}")
        logging.info(f"Found file: {path}")

    def load_all_sales(self) -> pd.DataFrame:
        """Load all_sales.csv into a DataFrame."""
        try:
            self._check_file_exists(self.all_sales_path)
            df = pd.read_csv(self.all_sales_path, parse_dates=['date'], low_memory=False)
            logging.info(f"Loaded all_sales.csv with shape {df.shape}")
            return df
        except Exception as e:
            raise CustomException(e)

    def load_product_hierarchy(self) -> pd.DataFrame:
        """Load product_hierarchy.csv into a DataFrame."""
        try:
            self._check_file_exists(self.product_hierarchy_path)
            df = pd.read_csv(self.product_hierarchy_path)
            logging.info(f"Loaded product_hierarchy.csv with shape {df.shape}")
            return df
        except Exception as e:
            raise CustomException(e)

    def load_store_cities(self) -> pd.DataFrame:
        """Load store_cities.csv into a DataFrame."""
        try:
            self._check_file_exists(self.store_cities_path)
            df = pd.read_csv(self.store_cities_path)
            logging.info(f"Loaded store_cities.csv with shape {df.shape}")
            return df
        except Exception as e:
            raise CustomException(e)

    def load_all_data(self):
        """Load all three datasets and return as a dictionary of DataFrames."""
        try:
            return {
                "all_sales": self.load_all_sales(),
                "product_hierarchy": self.load_product_hierarchy(),
                "store_cities": self.load_store_cities()
            }
        except Exception as e:
            raise CustomException(e)


# Example usage
if __name__ == "__main__":
    loader = DataLoader()
    try:
        data = loader.load_all_data()
        logging.info("All datasets loaded successfully.")
    except CustomException as ce:
        logging.error(f"Failed to load datasets: {ce}")