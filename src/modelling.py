import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import roc_auc_score, classification_report

from src.logger import logging
from src.exception import CustomException


class StoreSuccessModelTrainer:
    def __init__(self, X_train, y_train, X_test, y_test, feature_names):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.feature_names = feature_names

        self.best_log_reg = None
        self.best_rf = None

    def train_logistic_regression(self):
        """
        Train Logistic Regression with simple GridSearch.
        """
        try:
            logging.info("Training Logistic Regression with CV")

            param_grid = {
                "C": [0.1, 1.0, 10.0]
            }

            model = LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                random_state=42
            )

            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

            grid = GridSearchCV(
                model,
                param_grid,
                scoring="roc_auc",
                cv=cv,
                n_jobs=-1
            )

            grid.fit(self.X_train, self.y_train)

            self.best_log_reg = grid.best_estimator_

            logging.info(f"Best Logistic Regression params: {grid.best_params_}")
            return self.best_log_reg

        except Exception as e:
            raise CustomException(f"Logistic Regression training failed: {str(e)}")

    def train_random_forest(self):
        """
        Train Random Forest with simple GridSearch.
        """
        try:
            logging.info("Training Random Forest with CV")

            param_grid = {
                "n_estimators": [100, 200],
                "max_depth": [None, 5, 10],
                "min_samples_leaf": [1, 5]
            }

            model = RandomForestClassifier(
                random_state=42,
                class_weight="balanced"
            )

            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

            grid = GridSearchCV(
                model,
                param_grid,
                scoring="roc_auc",
                cv=cv,
                n_jobs=-1
            )

            grid.fit(self.X_train, self.y_train)

            self.best_rf = grid.best_estimator_

            logging.info(f"Best Random Forest params: {grid.best_params_}")
            return self.best_rf

        except Exception as e:
            raise CustomException(f"Random Forest training failed: {str(e)}")

    def evaluate(self, model, model_name: str):
        """
        Evaluate model on test set.
        """
        y_pred = model.predict(self.X_test)
        y_prob = model.predict_proba(self.X_test)[:, 1]

        auc = roc_auc_score(self.y_test, y_prob)

        print("\n" + "=" * 80)
        print(f"{model_name.upper()} PERFORMANCE (TEST SET)")
        print("=" * 80)

        print(f"ROC-AUC: {auc:.3f}")
        print("\nClassification Report:")
        print(classification_report(self.y_test, y_pred))

        return auc

    def explain_logistic_regression(self):
        """
        Explain drivers using coefficients.
        """
        coefs = pd.DataFrame({
            "feature": self.feature_names,
            "coefficient": self.best_log_reg.coef_[0]
        }).sort_values("coefficient", ascending=False)

        print("\nTop Positive Drivers:")
        print(coefs.head(10))

        print("\nTop Negative Drivers:")
        print(coefs.tail(10))

        return coefs

    def explain_random_forest(self):
        """
        Explain drivers using feature importance.
        """
        importance = pd.DataFrame({
            "feature": self.feature_names,
            "importance": self.best_rf.feature_importances_
        }).sort_values("importance", ascending=False)

        print("\nTop Drivers (Random Forest):")
        print(importance.head(10))

        return importance

# Logistic Regression
if __name__ == "__main__":
    from src.data_loader import DataLoader
    from src.data_cleaning import DataPreprocessor
    from src.eda import EDA
    from src.feature_engineering import FeatureEngineering
    from src.modelling import StoreSuccessModelTrainer

    # 1. Load raw data
    loader = DataLoader()
    raw_data = loader.load_all_data()

    # 2. Clean data
    preprocessor = DataPreprocessor(raw_data)
    cleaned_data = preprocessor.preprocess_all()

    # 3. Run EDA
    eda = EDA(
        cleaned_data['all_sales'],
        cleaned_data['store_cities']
    )
    store_data = eda.run_eda()

    # 4. Feature Engineering
    fe = FeatureEngineering(store_data)
    data = fe.prepare_data()  # returns dict with X_train, X_test, y_train, y_test, feature_names
    print("Feature engineering complete")

    # 5. Extract variables explicitly
    X_train = data['X_train']
    X_test = data['X_test']
    y_train = data['y_train']
    y_test = data['y_test']
    numerical_features = data['numerical_features']
    categorical_features = data['categorical_features']
    preprocessor = data['preprocessor']

    # Apply preprocessor
    X_train = preprocessor.fit_transform(X_train)
    X_test = preprocessor.transform(X_test)

    # Combine numeric + categorical for feature names
    feature_names = numerical_features + list(preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_features))

    # 6. Initialize model trainer
    trainer = StoreSuccessModelTrainer(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        feature_names=feature_names
    )

    # Logistic Regression
    log_reg = trainer.train_logistic_regression()
    trainer.evaluate(log_reg, "Logistic Regression")
    trainer.explain_logistic_regression()

    # Random Forest
    rf = trainer.train_random_forest()
    trainer.evaluate(rf, "Random Forest")
    trainer.explain_random_forest()

    new_stores = pd.DataFrame([
    {"store_size": 62, "store_radius_pop_density": 120.2, "store_radius_hh_income": 52600, "avg_product_price": 10.5, "storetype_id": "ST03"},
    {"store_size": 40,  "store_radius_pop_density": 520.6, "store_radius_hh_income": 69800, "avg_product_price": 8.7, "storetype_id": "ST04"},
    {"store_size": 50, "store_radius_pop_density": 303.4, "store_radius_hh_income": 78400, "avg_product_price": 7.9, "storetype_id": "ST01"}
    ])
    new_stores['income_to_price_ratio'] = new_stores['store_radius_hh_income'] / new_stores['avg_product_price']
    X_new = data['preprocessor'].transform(new_stores)
    log_reg = trainer.best_log_reg  # trained model with C=10

    # Class prediction
    predictions = log_reg.predict(X_new)

    # Probability of success
    probabilities = log_reg.predict_proba(X_new)[:, 1]

    for i, (pred, prob) in enumerate(zip(predictions, probabilities), 1):
        status = "Successful" if pred == 1 else "Not Successful"
        print(f"Store {i}: {status} (probability of success = {prob:.2f})")