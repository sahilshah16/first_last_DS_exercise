import duckdb
import pandas as pd
from src.logger import logging
from src.exception import CustomException


def q1_largest_product_revenue_2018(cleanedData: dict):
    """
    Q1:
    Find the largest product (by product_volume) and
    calculate how much revenue it earned in 2018.

    Args:
        cleaned_data (dict): Output from DataPreprocessor.preprocess_all()

    Returns:
        pd.DataFrame: product_id, product_volume, total_revenue_2018
    """
    try:
        logging.info("Starting Q1: Largest product revenue in 2018")

        all_sales = cleanedData["all_sales"]
        product_hierarchy = cleanedData["product_hierarchy"]

        # Register DataFrames as DuckDB tables
        con = duckdb.connect()
        con.register("all_sales", all_sales)
        con.register("product_hierarchy", product_hierarchy)

        query = """
        WITH largest_product AS (
            SELECT product_id, product_volume FROM product_hierarchy
            WHERE product_volume = (
                SELECT MAX(product_volume)
                FROM product_hierarchy
            )
        )
        SELECT lp.product_id, lp.product_volume, SUM(s.revenue) AS total_revenue_2018 FROM largest_product lp
        INNER JOIN all_sales s ON lp.product_id = s.product_id
        WHERE s.year = 2018
        GROUP BY lp.product_id, lp.product_volume
        """

        result = con.execute(query).df()

        logging.info(
            f"Q1 completed successfully. Result: {result.to_dict(orient='records')}"
        )

        return result

    except Exception as e:
        raise CustomException(f"Error executing Q1: {str(e)}")

def q2_city_largest_revenue_2018(cleanedData: dict):
    """
    Q2:
    Find the city produced that had the most revenue in 2018 and
    how much revenue was it

    Args:
        cleaned_data (dict): Output from DataPreprocessor.preprocess_all()

    Returns:
        pd.DataFrame: city, total_revenue 
    """
    try:
        logging.info("Starting Q2: City with largest revenue in 2019")

        all_sales = cleanedData["all_sales"]
        store_cities = cleanedData["store_cities"]

        # Register DataFrames as DuckDB tables
        con = duckdb.connect()
        con.register("all_sales", all_sales)
        con.register("store_cities", store_cities)

        query = """
        WITH city_revenue AS (
            SELECT sc.city_id, SUM(s.revenue) AS total_revenue FROM all_sales s
            INNER JOIN store_cities sc ON s.store_id = sc.store_id
            WHERE s.year = 2018
            GROUP BY sc.city_id
        )
        SELECT city_id, total_revenue FROM city_revenue
        WHERE total_revenue = (
            SELECT MAX(total_revenue) FROM city_revenue
        );
        """

        result = con.execute(query).df()

        logging.info(
            f"Q2 completed successfully. Result: {result.to_dict(orient='records')}"
        )

        return result

    except Exception as e:
        raise CustomException(f"Error executing Q2: {str(e)}")

def q3_product_greatest_number_days_not_in_stock_2019(cleanedData: dict):
    """
    Q3:
        Find product that had the greatest number of days in 2019 where 
        it was not available in at least one store and for how many days

    Args:
        cleaned_data (dict): Output from DataPreprocessor.preprocess_all()

    Returns:
        pd.DataFrame: product_id, days_not_available_in_at_least_one_store 
    """
    try:
        logging.info("Starting Q3: Product greatest")

        all_sales = cleanedData["all_sales"]

        # Register DataFrames as DuckDB tables
        con = duckdb.connect()
        con.register("all_sales", all_sales)

        query = """
        WITH product_day_stock AS (
            SELECT
                product_id,
                date,
                MIN(stock) AS min_stock
            FROM all_sales
            WHERE year = 2019
            GROUP BY product_id, date
        ),
        out_of_stock_days AS (
            SELECT
                product_id,
                COUNT(*) AS out_of_stock_days
            FROM product_day_stock
            WHERE min_stock <= 0
            GROUP BY product_id
        )
        SELECT
            product_id,
            out_of_stock_days
        FROM out_of_stock_days
        WHERE out_of_stock_days = (
            SELECT MAX(out_of_stock_days) FROM out_of_stock_days
        );
        """

        result = con.execute(query).df()

        logging.info(
            f"Q3 completed successfully. Result: {result.to_dict(orient='records')}"
        )

        return result

    except Exception as e:
        raise CustomException(f"Error executing Q2: {str(e)}")


def execute_all_queries(cleanedData: dict):
    """
    Execute all SQL queries (Q1-Q3) and return results.
    
    Returns:
        Dict containing results for all three queries
    """
    try:
        logging.info("Executing all SQL queries")
        
        results = {
            'Q1': q1_largest_product_revenue_2018(cleanedData),
            'Q2': q2_city_largest_revenue_2018(cleanedData),
            'Q3': q3_product_greatest_number_days_not_in_stock_2019(cleanedData)
        }
        
        logging.info("All SQL queries completed successfully")
        return results
        
    except Exception as e:
        raise CustomException(f"Error executing queries: {str(e)}")

if __name__ == "__main__":
    from src.data_loader import DataLoader
    from src.data_cleaning import DataPreprocessor
    
    # Load and preprocess data
    loader = DataLoader()
    raw_data = loader.load_all_data()

    preprocessor = DataPreprocessor(raw_data)
    cleaned_data = preprocessor.preprocess_all()

    q1_result = execute_all_queries(cleaned_data)
    print(q1_result)
        