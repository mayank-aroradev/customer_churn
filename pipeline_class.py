
import pickle
from pathlib import Path

import numpy as np
import pandas as pd


class PredictorPipeline:
    def __init__(
        self,
        catboost_model,
        pca_transformer,
        scaler_transformer,
        top_15_categories,
        top_states,
        top_payments,
        black_friday_dates,
        columns_order,
    ):
        self.model = catboost_model
        self.pca = pca_transformer
        self.scaler = scaler_transformer
        self.top_15_categories = top_15_categories
        self.top_states = top_states
        self.top_payments = top_payments
        self.black_friday_dates = black_friday_dates
        self.columns_order = columns_order

    def preprocess(self, raw_df):
        df = raw_df.copy()

        # Convert timestamps
        df["order_purchase_timestamp"] = pd.to_datetime(
            df["order_purchase_timestamp"]
        )
        df["order_delivered_customer_date"] = pd.to_datetime(
            df["order_delivered_customer_date"], errors="coerce"
        )
        df["order_estimated_delivery_date"] = pd.to_datetime(
            df["order_estimated_delivery_date"], errors="coerce"
        )

        # Delivery delay
        df["delivery_delay_days"] = (
            df["order_delivered_customer_date"]
            - df["order_estimated_delivery_date"]
        ).dt.days
        df["delivery_delay_days"] = df["delivery_delay_days"].fillna(-12.0)

        # Freight ratio
        df["freight_ratio"] = np.where(
            df["price"] > 0,
            df["freight_value"] / df["price"],
            0.0,
        )

        # Review features
        df["has_review"] = df["review_score"].notnull().astype(int)
        df["review_score"] = df["review_score"].fillna(5.0)

        # Value per item
        df["value_per_item"] = df["price"] / (df["total_items"] + 1e-5)

        # Time-based features
        df["order_purchase_hour"] = df[
            "order_purchase_timestamp"
        ].dt.hour
        df["order_purchase_dayofweek"] = df[
            "order_purchase_timestamp"
        ].dt.dayofweek
        df["order_purchase_month"] = df[
            "order_purchase_timestamp"
        ].dt.month

        # Categorical cleaning
        df["product_category_clean"] = df[
            "product_category_name"
        ].apply(
            lambda x: x if x in self.top_15_categories else "other"
        )

        df["customer_state_clean"] = df["customer_state"].apply(
            lambda x: x if x in self.top_states else "other"
        )

        df["payment_type_clean"] = df["payment_type"].fillna(
            "unknown"
        ).apply(
            lambda x: x if x in self.top_payments else "other"
        )

        # State match
        df["state_match"] = (
            df["customer_state"] == df["seller_state"]
        ).astype(int)

        # Black Friday indicator
        df["order_purchase_date"] = (
            df["order_purchase_timestamp"].dt.date
        )

        df["ordered_on_black_friday"] = df.apply(
            lambda row: int(
                row["order_purchase_timestamp"].year
                in self.black_friday_dates
                and row["order_purchase_date"]
                == self.black_friday_dates[
                    row["order_purchase_timestamp"].year
                ]
            ),
            axis=1,
        )

        # One-hot encoding
        categorical_cols = [
            "product_category_clean",
            "customer_state_clean",
            "payment_type_clean",
        ]

        encoded_cats = pd.get_dummies(
            df[categorical_cols],
            drop_first=True,
            dtype=int,
        )

        # Numerical features
        numerical_cols = [
            "price",
            "freight_value",
            "total_items",
            "payment_value",
            "review_score",
            "delivery_delay_days",
            "freight_ratio",
            "has_review",
            "value_per_item",
            "order_purchase_hour",
            "order_purchase_dayofweek",
            "order_purchase_month",
            "state_match",
            "ordered_on_black_friday",
            "geolocation_lat",
            "geolocation_lng",
        ]

        processed_df = pd.concat(
            [df[numerical_cols], encoded_cats],
            axis=1,
        )

        # Match the training column order
        processed_df = processed_df.reindex(
            columns=self.columns_order,
            fill_value=0,
        )

        # Scaling and PCA
        scaled_features = self.scaler.transform(processed_df)
        pca_features = self.pca.transform(scaled_features)

        pca_cols = [
            f"PC{i}"
            for i in range(1, pca_features.shape[1] + 1)
        ]

        return pd.DataFrame(
            pca_features,
            columns=pca_cols,
            index=df.index,
        )

    def predict(self, raw_df):
        pca_df = self.preprocess(raw_df)

        predictions = self.model.predict(pca_df)
        probabilities = self.model.predict_proba(pca_df)

        return predictions, probabilities

