"""
Configuration file for Pawsible Apparel DS Exercise.

This file contains all file paths, constants, and parameters used
across the project to ensure consistency, reproducibility, and maintainability.
"""

# ----------------------------
# File paths
# ----------------------------
ALL_SALES_CSV = "data/raw/all_sales.csv"
PRODUCT_HIERARCHY_CSV = "data/raw/product_hierarchy.csv"
STORE_CITIES_CSV = "data/raw/store_cities.csv"

PROCESSED_DATA_DIR = "data/processed/"

# ----------------------------
# Analysis years
# ----------------------------
YEAR_2018 = 2018
YEAR_2019 = 2019

# ----------------------------
# Stock & sales constants
# ----------------------------
OUT_OF_STOCK_THRESHOLD = 0  # Defines out-of-stock condition

# ----------------------------
# Database config (if needed)
# ----------------------------
DATABASE_NAME = "pawsible.db"

# ----------------------------
# Random seed for reproducibility
# ----------------------------
RANDOM_STATE = 42