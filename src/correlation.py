import duckdb
from scipy import stats
import pandas as pd
from src.exception import CustomException
from src.logger import logging


    
def get_store_sales_vs_size_dataset(cleanedData: dict):
    """
    Prepare analytical dataset for Q4.

    Returns:
        pd.DataFrame with:
        - store_id
        - year
        - total_units_sold
        - store_size
    """
    try:
        logging.info("Preparing Q4 analytical dataset (SQL only)")
        
        all_sales = cleanedData["all_sales"]
        store_cities = cleanedData["store_cities"]

        # Register DataFrames as DuckDB tables
        con = duckdb.connect()
        con.register("all_sales", all_sales)
        con.register("store_cities", store_cities)
        query = """
        SELECT
        s.store_id,
        s.year,
        SUM(s.sales) AS total_units_sold,
        sc.store_size
    FROM all_sales s
    JOIN store_cities sc
        ON s.store_id = sc.store_id
    GROUP BY
        s.store_id,
        s.year,
        sc.store_size;
        """

        df = con.execute(query).df()

        logging.info("Q4 dataset prepared: %d rows", len(df))
        return df

    except Exception as e:
        logging.error("Failed to prepare Q4 dataset", exc_info=True)
        raise CustomException(f"Q4 dataset error: {str(e)}")

def analyse_store_size_vs_sales(cleanedData: dict) -> dict:
    """
    Perform correlation analysis between store size and total units sold.

    Steps:
    - Prepare analytical dataset using SQL
    - Compute overall Pearson & Spearman correlations
    - Compute year-by-year Pearson correlations

    Returns:
        dict containing:
            - overall_correlation
            - yearly_correlation (DataFrame)
    """
    try:
        logging.info("Starting Q4 correlation analysis")

        # Prepare analytical dataset
        store_data = get_store_sales_vs_size_dataset(cleanedData)

        # Select required columns & drop any missing values (defensive)
        analysis_data = store_data[
            ['store_size', 'total_units_sold', 'year']
        ].dropna()

        logging.info("Analysis dataset prepared with %d rows", len(analysis_data))

        # ---------------------------
        # Overall correlations
        # ---------------------------
        pearson_corr, pearson_pval = stats.pearsonr(
            analysis_data['store_size'],
            analysis_data['total_units_sold']
        )

        spearman_corr, spearman_pval = stats.spearmanr(
            analysis_data['store_size'],
            analysis_data['total_units_sold']
        )

        overall_results = {
            "pearson_correlation": pearson_corr,
            "pearson_p_value": pearson_pval,
            "spearman_correlation": spearman_corr,
            "spearman_p_value": spearman_pval,
            "sample_size": len(analysis_data)
        }

        logging.info(
            "Overall correlations calculated (Pearson=%.4f, Spearman=%.4f)",
            pearson_corr,
            spearman_corr
        )

        # ---------------------------
        # Year-by-year correlation
        # ---------------------------
        yearly_corr = []

        for year in sorted(analysis_data['year'].unique()):
            year_data = analysis_data[analysis_data['year'] == year]

            # Require sufficient observations for stability
            if len(year_data) > 2:
                corr, pval = stats.pearsonr(
                    year_data['store_size'],
                    year_data['total_units_sold']
                )

                yearly_corr.append({
                    "year": int(year),
                    "correlation": corr,
                    "p_value": pval,
                    "sample_size": len(year_data)
                })

        yearly_corr_df = pd.DataFrame(yearly_corr)

        logging.info(
            "Year-by-year correlation completed for %d years",
            len(yearly_corr_df)
        )

        return {
            "overall_correlation": overall_results,
            "yearly_correlation": yearly_corr_df
        }

    except Exception as e:
        logging.error("Q4 correlation analysis failed", exc_info=True)
        raise CustomException(f"Q4 correlation analysis error: {str(e)}")

if __name__ == "__main__":
    from src.data_loader import DataLoader
    from src.data_cleaning import DataPreprocessor
    
    
    # Load and preprocess data
    loader = DataLoader()
    raw_data = loader.load_all_data()

    preprocessor = DataPreprocessor(raw_data)
    cleaned_data = preprocessor.preprocess_all()
    
    q1_result = analyse_store_size_vs_sales(cleaned_data)
    print(q1_result)