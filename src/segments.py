"""
Q5: Product Segmentation

Segment products based on physical dimensions using K-Means clustering.


"""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from typing import Dict
from src.logger import logging
from src.exception import CustomException
from src.config import RANDOM_STATE




class Segmentation:
    """
    Q5: Segment products based on physical dimensions using K-Means clustering
    """

    def __init__(self, product_hierarchy: pd.DataFrame):
        """
        Args:
            product_hierarchy (pd.DataFrame): Cleaned product hierarchy data
        """
        self.product_data = product_hierarchy.copy()

        self.feature_cols = [
            "product_length",
            "product_width",
            "product_depth",
            "product_volume"
        ]

        self.scaler = StandardScaler()
        self.optimal_k = None
        self.model = None

    def prepare_features(self) -> np.ndarray:
        """Prepare and scale features for clustering."""
        try:
            logging.info("Preparing features for product segmentation")

            features_scaled = self.scaler.fit_transform(
                self.product_data[self.feature_cols]
            )

            logging.info(
                f"Prepared {len(self.product_data)} products "
                f"with {len(self.feature_cols)} features"
            )

            return features_scaled

        except Exception as e:
            raise CustomException(f"Error preparing features: {str(e)}")

    def find_optimal_clusters(self, features_scaled: np.ndarray, max_k: int = 10) -> dict:
        """Identify optimal number of clusters using silhouette score."""
        try:
            logging.info(f"Finding optimal number of clusters (k=2 to {max_k})")

            silhouette_scores = []
            k_range = range(2, max_k + 1)

            for k in k_range:
                kmeans = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
                labels = kmeans.fit_predict(features_scaled)
                score = silhouette_score(features_scaled, labels)
                silhouette_scores.append(score)

            optimal_idx = int(np.argmax(silhouette_scores))
            self.optimal_k = list(k_range)[optimal_idx]

            results = {
                "k_values": list(k_range),
                "silhouette_scores": silhouette_scores,
                "optimal_k": self.optimal_k,
                "optimal_silhouette": silhouette_scores[optimal_idx]
            }

            logging.info(
                f"Optimal k = {self.optimal_k} "
                f"(silhouette score = {results['optimal_silhouette']:.4f})"
            )

            return results

        except Exception as e:
            raise CustomException(f"Error finding optimal clusters: {str(e)}")

    def q5_segment_products(self, n_clusters: int = None):
        """
        Q5: Segment products based on dimensions using K-Means clustering.

        Args:
            n_clusters: Optional number of clusters (overrides optimal k)

        Returns:
            Dict containing segmentation results and analysis
        """
        try:
            logging.info("Executing Q5: Product segmentation")

            # Prepare features
            features_scaled = self.prepare_features()

            # Use provided n_clusters or compute optimal
            if n_clusters is not None:
                self.optimal_k = n_clusters
                optimization_results = None
            elif self.optimal_k is None:
                optimization_results = self.find_optimal_clusters(features_scaled)
                self.optimal_k = optimization_results['optimal_k']
            else:
                optimization_results = None

            # Train final model
            self.model = KMeans(n_clusters=self.optimal_k, random_state=RANDOM_STATE, n_init=10)
            self.product_data['segment'] = self.model.fit_predict(features_scaled)

            result = {
                'n_clusters': self.optimal_k,
                'optimization_results': optimization_results,
                'silhouette_score': silhouette_score(features_scaled, self.product_data['segment'])
            }

            logging.info(f"Created {self.optimal_k} product segments")
            return result

        except Exception as e:
            raise CustomException(f"Error in product segmentation: {str(e)}")

    def _calculate_segment_statistics(self) -> pd.DataFrame:
        """Calculate statistics for each cluster/segment."""
        try:
            stats = self.product_data.groupby('segment').agg({
                'product_id': 'count',
                'product_length': ['mean', 'std', 'min', 'max'],
                'product_depth': ['mean', 'std', 'min', 'max'],
                'product_width': ['mean', 'std', 'min', 'max'],
                'product_volume': ['mean', 'std', 'min', 'max']
            }).reset_index()

            # Flatten MultiIndex columns
            stats.columns = ['_'.join(col).strip('_') for col in stats.columns.values]
            return stats

        except Exception as e:
            raise CustomException(f"Error calculating segment statistics: {str(e)}")

    def _assign_segment_names(self) -> Dict[int, str]:
        """Assign meaningful names to product segments based on volume and shape."""
        try:
            segment_stats = self._calculate_segment_statistics()
            names = {}

            # Sort segments by average volume
            segment_stats = segment_stats.sort_values('product_volume_mean')
            n_segments = len(segment_stats)

            for idx, (_, row) in enumerate(segment_stats.iterrows()):
                segment_id = int(row['segment'])
                
                # Determine size category
                if n_segments <= 3:
                    if idx == 0:
                        size = "Small"
                    elif idx == n_segments - 1:
                        size = "Large"
                    else:
                        size = "Medium"
                elif n_segments <= 5:
                    size_categories = ["Extra Small", "Small", "Medium", "Large", "Extra Large"]
                    size = size_categories[idx]
                else:
                    percentile = (idx + 1) / n_segments * 100
                    if percentile <= 20:
                        size = "Extra Small"
                    elif percentile <= 40:
                        size = "Small"
                    elif percentile <= 60:
                        size = "Medium"
                    elif percentile <= 80:
                        size = "Large"
                    else:
                        size = "Extra Large"

                # Determine shape
                avg_dims = [row['product_length_mean'], row['product_depth_mean'], row['product_width_mean']]
                max_dim = max(avg_dims)
                min_dim = min(avg_dims)

                if max_dim / min_dim > 2.5:
                    shape = "Elongated"
                elif max_dim / min_dim > 1.5:
                    shape = "Rectangular"
                else:
                    shape = "Compact"

                names[segment_id] = f"{size} & {shape} Products"

            # Add to product_data
            self.product_data['segment_name'] = self.product_data['segment'].map(names)

            logging.info(f"Segment names assigned: {names}")
            return names

        except Exception as e:
            raise CustomException(f"Error assigning segment names: {str(e)}")


if __name__ == "__main__":
    from src.data_loader import DataLoader
    from src.data_cleaning import DataPreprocessor
    
    # Load and preprocess data
    loader = DataLoader()
    raw_data = loader.load_all_data()

    preprocessor = DataPreprocessor(raw_data)
    cleaned_data = preprocessor.preprocess_all()

    # Run segmentation
    seg = Segmentation(cleaned_data['product_hierarchy'])
    seg.q5_segment_products()  # perform KMeans clustering
    segment_names = seg._assign_segment_names()  # assign meaningful names
    
    print("\nSegment Names:")
    print(segment_names)
    print("\nSample Products:")
    print(seg.product_data[['product_id', 'segment', 'segment_name']].head(10))