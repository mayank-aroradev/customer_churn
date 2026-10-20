import streamlit as st
import requests
import datetime
from predictor import predict_output

st.title("Customer Churn Predictor App")
st.markdown("Enter your details below:")

purchase_date = st.date_input("Order Purchase Date", value=datetime.date(2018, 5, 10))
purchase_time = st.time_input("Order Purchase Time", value=datetime.time(10, 0))
order_purchase_timestamp = datetime.datetime.combine(purchase_date, purchase_time)

delivery_date = st.date_input("Order Delivered to Customer Date", value=datetime.date(2018, 5, 12))
delivery_time = st.time_input("Order Delivered to Customer Time", value=datetime.time(12, 0))
order_delivered_customer_date = datetime.datetime.combine(delivery_date, delivery_time)

estimated_date = st.date_input("Order Estimated Delivery Date", value=datetime.date(2018, 5, 27))
order_estimated_delivery_date = datetime.datetime.combine(estimated_date, datetime.time(0, 0))

price = st.number_input("Price (BRL)", min_value=0.0, value=1000.0)
freight_value = st.number_input("Freight Value (BRL)", min_value=0.0, value=100.0)
total_items = st.number_input("Total Items Purchased", min_value=1, value=10)
payment_value = st.number_input("Total Payment Value (BRL)", min_value=0.0, value=1000.0)
review_score = st.number_input("Review Score", min_value=1.0, max_value=5.0, value=5.0)
geolocation_lat = st.number_input("Customer Latitude", min_value=-45.0, max_value=10.0, value=-23.55)
geolocation_lng = st.number_input("Customer Longitude", min_value=-75.0, max_value=-30.0, value=-46.63)

product_category_name = st.text_input("Product Category Name", value="beleza_saude")
customer_state = st.text_input("Customer State (2-letter code)", value="SP", max_chars=2)
seller_state = st.text_input("Seller State (2-letter code)", value="SP", max_chars=2)
payment_type = st.selectbox(
    "Payment Type",
    options=['credit_card', 'boleto', 'voucher', 'debit_card', 'unknown']
)

if st.button("prediction premium category"):
    input_data = {
        'order_purchase_timestamp': order_purchase_timestamp,
        'order_delivered_customer_date': order_delivered_customer_date,
        'order_estimated_delivery_date': order_estimated_delivery_date,
        'price': price,
        'freight_value': freight_value,
        'total_items': total_items,
        'payment_value': payment_value,
        'review_score': review_score,
        'product_category_name': product_category_name,
        'customer_state': customer_state,
        'seller_state': seller_state,
        'payment_type': payment_type,
        'geolocation_lat': geolocation_lat,
        'geolocation_lng': geolocation_lng
    }
    
    # Convert dictionary to DataFrame for the pipeline
    import pandas as pd
    input_df = pd.DataFrame([input_data])
    
    # Run inference using the loaded pipeline
    pred_class, pred_proba = predict_output(input_df)
    
    # Display output
    if pred_class[0] == 1:
        st.success(f"Likely to Reorder! (Probability: {pred_proba[0][1]:.2%})")
    else:
        st.warning(f"Unlikely to Reorder. (Probability of Reordering: {pred_proba[0][1]:.2%})")