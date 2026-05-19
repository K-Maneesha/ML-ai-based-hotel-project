import streamlit as st
import numpy as np
import joblib
import pandas as pd

st.set_page_config(
    page_title="Hotel Prediction System",
    page_icon="🏨",
    layout="wide"
)

st.markdown(
    "<h1 style='text-align:center;color:#2c3e50;'>🏨 Hotel ML Prediction Dashboard</h1>",
    unsafe_allow_html=True
)

demand_model = joblib.load("m2_demand_model.pkl")
price_model = joblib.load("m3_price_model.pkl")
sentiment_model = joblib.load("m4_sentiment_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

months = [
    "January","February","March","April","May","June",
    "July","August","September","October","November","December"
]

menu = st.sidebar.radio(
    "Select Module",
    [
        "Room Demand Prediction",
        "Seasonal Pricing Prediction",
        "Sentiment Analysis"
    ]
)

# ==========================================================
# ROOM DEMAND PREDICTION
# ==========================================================
if menu == "Room Demand Prediction":

    st.subheader("📊 Room Demand Prediction")

    month = st.selectbox("Month", months)
    guests = st.number_input("Guests", 1, 10, 2)
    stay_nights = st.slider("Stay Nights", 1, 14, 2)

    room_type = st.selectbox(
        "Room Type",
        ["Standard", "Deluxe", "Premium"]
    )

    lead_time = st.slider("Lead Time", 0, 365, 30)

    month_num = months.index(month) + 1

    room_map = {
        "Standard": 1,
        "Deluxe": 2,
        "Premium": 3
    }

    room_num = room_map[room_type]

    input_data = np.array([[
        month_num,
        guests,
        stay_nights,
        room_num,
        lead_time
    ]])

    if st.button("Predict Demand"):

        demand_level = int(demand_model.predict(input_data)[0])

        if demand_level == 0:
            label = "Low Demand"
            rooms = "40 - 70 rooms/day"
            color = "red"

        elif demand_level == 1:
            label = "Medium Demand"
            rooms = "70 - 110 rooms/day"
            color = "orange"

        else:
            label = "High Demand"
            rooms = "110 - 160 rooms/day"
            color = "green"

        st.markdown(
            f"""
            <div style="
            background:white;
            padding:15px;
            border-radius:10px;
            font-size:22px;
            font-weight:bold;
            color:{color};">
            Demand Level: {label}<br>
            Expected Rooms Booked: {rooms}
            </div>
            """,
            unsafe_allow_html=True
        )

# ==========================================================
# SEASONAL PRICING PREDICTION
# ==========================================================
elif menu == "Seasonal Pricing Prediction":

    st.subheader("💰 Seasonal Pricing Prediction")

    month = st.selectbox("Month", months)
    guests = st.number_input("Guests", 1, 10, 2)
    stay_nights = st.slider("Stay Nights", 1, 14, 2)

    room_type = st.selectbox(
        "Room Type",
        ["Standard", "Deluxe", "Premium"]
    )

    lead_time = st.slider("Lead Time", 0, 365, 30)

    month_num = months.index(month) + 1

    input_data = np.array([[
        month_num,
        month_num * 4,
        stay_nights,
        guests,
        0
    ]])

    if st.button("Predict Price"):

        price_inr = price_model.predict(input_data)[0]

        price_inr = max(3000, min(price_inr, 20000))

        if room_type == "Standard":
            price_inr = price_inr * 0.9
        elif room_type == "Deluxe":
            price_inr = price_inr * 1.1
        else:
            price_inr = price_inr * 1.3

        st.success(
            f"Predicted Room Price per Night: ₹ {round(price_inr, 2)}"
        )

        st.info(
            "This price represents the average price for one room per night."
        )

# ==========================================================
# SENTIMENT ANALYSIS
# ==========================================================
else:

    st.subheader("🧠 Sentiment Analysis")

    df = pd.read_csv("tripadvisor_hotel_reviews.csv")

    def label(r):
        if r >= 4:
            return "Positive"
        elif r == 3:
            return "Neutral"
        else:
            return "Negative"

    df["Sentiment"] = df["Rating"].apply(label)

    review = st.text_area("Enter Customer Review")

    if st.button("Analyze"):

        vec = vectorizer.transform([review])
        pred = sentiment_model.predict(vec)[0]

        if pred == "Positive":
            st.success("Positive 😊")
        elif pred == "Neutral":
            st.warning("Neutral 😐")
        else:
            st.error("Negative 😠")

    st.markdown("---")

    option = st.radio(
        "Filter Dataset",
        ["All", "Positive", "Neutral", "Negative"]
    )

    if option == "All":
        data = df
    else:
        data = df[df["Sentiment"] == option]

    st.write("Review Count:", data.shape[0])
    st.dataframe(data[["Review", "Sentiment"]].head(50))