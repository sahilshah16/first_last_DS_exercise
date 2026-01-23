# Pawsible Apparel - Data Science Analysis

## Setup

1. **Create virtual environment** (optional but recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Add data files** to `data/raw` folder:
- `all_sales.csv`
- `product_hierarchy.csv`
- `store_cities.csv`

## Running the Analysis

Run each question from the project root folder:

**Q1-Q3: SQL Queries**
```bash
python -m src.sql_queries
```

**Q4: Store Size Correlation Analysis**
```bash
python -m src.correlation
```

**Q5: Product Segmentation**
```bash
python -m src.segmentation
```

**Q6: Store Success Prediction**
```bash
python -m src.modelling
```

## Outputs

- Detailed logs are available in the `logs/` folder

## Project Structure

```
pawsible_apparel/
├── data/              # CSV data files
├── src/               # Source code
├── plots/             # EDA results
├── logs/              # Execution logs
└── requirements.txt   # Python dependencies
```
**Q1 Results:** Largest product by dimensions is **P0621** (volume: 1,600,000 cubic units), which generated **$40,732.23** in revenue during 2018.

**Q2 Results:** City **C014** produced the highest revenue in 2018 with **$4,448,620.88** in total sales.

**Q3 Results:** Product **P0185** had the most stockout days in 2019, being unavailable in at least one store for **363 days** (99.5% of the year). This indicates severe inventory management issues requiring immediate attention.

**Q4 Results:** Strong **negative correlation** between store size and sales (Pearson: -0.51, p < 0.001). Smaller stores actually generate more sales than larger stores. This relationship is statistically significant and consistent across all years (2017-2019), with correlation strengthening over time (-0.48 → -0.57). Spearman correlation (-0.73) suggests a strong monotonic relationship. **Key insight:** Store size is NOT a positive driver of sales; smaller, more efficient stores outperform larger ones.

**Q5 Results:** K-Means clustering identified **2 optimal product segments** (silhouette score: 0.86, indicating excellent cluster separation). 

**Segments:**
- **Segment 0:** Small & Elongated Products
- **Segment 1:** Large & Rectangular Products

**Recommendation:** Target promotions differently by segment - compact packaging/storage promotions for small elongated items, different messaging for large rectangular products. High silhouette score confirms these are distinct, well-separated groups ideal for customised marketing.

**Q6 Results:** Built classification models to predict store success (top 33% by revenue).

**a) Main Drivers of Success:**
1. **Store radius population density** (strongest driver, correlation: 0.83)
2. **Average product price** (correlation: 0.66)
3. **Store radius household income** (correlation: 0.33)
4. **Store size** (negative correlation: -0.57, smaller stores perform better)
5. **Store type** (ST01 and ST04 outperform ST02 and ST03)

**b) Model Performance:**
- **Logistic Regression:** ROC-AUC = 1.00, Accuracy = 100% (perfect classification on test set)
- **Random Forest:** ROC-AUC = 1.00, Accuracy = 97%
- Both models show excellent performance with high precision and recall
- Consistent feature importance across models validates findings

**c) New Store Recommendations:**
- **Store 1:** **NOT RECOMMENDED** (0% success probability)
- **Store 2:** **STRONGLY RECOMMEND** (100% success probability)
- **Store 3:** **RECOMMEND** (68% success probability)

**Key Insight:** Success depends heavily on location demographics (population density + income) rather than store size. Focus expansion in dense, moderate-income areas with premium product pricing.
