# Customer Reorder Predictor

A machine learning app that estimates the probability that a customer will place a **second order within 180 days** of their first one, built on the [Olist Brazilian E-Commerce dataset](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce).

**Live demo:** https://customerchurn-6jcp5bsdcvnh6pmykmvdbd.streamlit.app/

> Free Streamlit apps sleep after inactivity. If you see a wake-up screen, click the button and give it a few seconds.

<!-- Add a screenshot of the app here:
![App screenshot](screenshot.png)
-->

## Problem

Repeat customers are far more valuable than one-time buyers, but most e-commerce customers never come back. In this dataset only about **3%** of customers place a second order within 180 days, which makes this a highly imbalanced prediction problem.

The app takes details of a customer's first order and returns:

- the predicted class (will reorder / will not reorder)
- the probability of reordering

## Approach

1. **Data:** merged the Olist orders, customers, items, payments, reviews, products and geolocation tables.
2. **Target:** `reordered_180`, which is 1 if the customer's second order came within 180 days of their first. Customers whose first order was within 180 days of the end of the data were dropped, so every customer has a full observation window.
3. **Features:** order value, freight, number of items, payment value and type, review score, delivery delay, purchase hour / day / month, top product categories, customer state, seller-customer state match, a Black Friday indicator and location.
4. **Models compared:** Random Forest, Logistic Regression, LightGBM (including tuned) and CatBoost.
5. **Final model:** CatBoost trained on **PCA components** (31 components retaining 95% of variance) after standardisation.
6. **Packaging:** all preprocessing and the model are wrapped in a single `PredictorPipeline` class, saved with pickle, so the app can take raw user input and return a prediction.

## Results

| Model | Setup | ROC AUC |
|---|---|---|
| Random Forest | balanced sample | about 0.51 to 0.55 |
| Logistic Regression | balanced sample | 0.567 |
| LightGBM (tuned) | balanced sample | 0.579 |
| LightGBM | full imbalanced data | 0.593 |
| **CatBoost + PCA** | full imbalanced data | **0.598** |

Predicting repeat purchases from first-order data alone is genuinely hard, and the final ROC AUC of about 0.60 reflects that. The model is best read as a rough propensity score rather than a precise forecast. Possible next steps are RFM-style features, customer behaviour before the first purchase, and probability calibration.

## Project structure

```
customer_churn/
├── connector.py        # Streamlit UI
├── predictor.py        # loads the saved pipeline and exposes predict_output()
├── pipeline_class.py   # PredictorPipeline class (needed to unpickle the model)
├── model.pkl           # trained pipeline
├── requirements.txt    # dependencies
└── README.md
```

## Run locally

```bash
git clone https://github.com/mayank-aroradev/customer_churn.git
cd customer_churn
python -m pip install -r requirements.txt
python -m streamlit run connector.py
```

The saved model was trained with specific library versions. If loading fails, install the versions pinned in `requirements.txt`.

## Tech stack

Python, pandas, NumPy, scikit-learn, CatBoost, LightGBM, Streamlit

## Author

**Mayank**, [GitHub](https://github.com/mayank-aroradev)

## Acknowledgements

Dataset: Olist, *Brazilian E-Commerce Public Dataset by Olist*, on Kaggle.
