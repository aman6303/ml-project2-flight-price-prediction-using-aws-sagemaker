import os
import pickle

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import xgboost as xgb

from mypackage.myclasses import *
from mypackage.mymethods import *

sklearn.set_config(
    transform_output="pandas"
)  # we need to give this in order to maintain the sequence of preprocessor transformer

# # web application
# st.set_page_config(page_title="Flights Prices Prediction", page_icon="✈️", layout="wide")

# st.title("Flights Prices Prediction - AWS SageMaker")

# # user inputs
# airline = st.selectbox(
#     "Airline:",
#     options=[
#         "Jet Airways",
#         "Indigo",
#         "Air India",
#         "Multiple Carriers",
#         "Spicejet",
#         "Vistara",
#         "Air Asia",
#         "Goair",
#     ],
# )

# doj = st.date_input("Date of Journey:")

# source = st.selectbox(
#     "Source", options=["Mumbai", "Delhi", "Kolkata", "Banglore", "Chennai"]
# )

# destination = st.selectbox(
#     "Destination",
#     options=["Hyderabad", "Cochin", "Banglore", "Delhi", "New Delhi", "Kolkata"],
# )

# dep_time = st.time_input("Departure Time:")

# arrival_time = st.time_input("Arrival Time:")

# duration = st.number_input("Duration (mins):", step=1)

# total_stops = st.number_input("Total Stops:", step=1, min_value=0)

# additional_info = st.selectbox(
#     "Additional Info:",
#     options=[
#         "No Info",
#         "In-Flight Meal Not Included",
#         "No Check-In Baggage Included",
#         "1 Long Layover",
#         "Change Airports",
#         "Business Class",
#         "Red-Eye Flight",
#     ],
# )

# x_new = pd.DataFrame(
#     dict(
#         airline=[airline],
#         date_of_journey=[doj],
#         source=[source],
#         destination=[destination],
#         dep_time=[dep_time],
#         arrival_time=[arrival_time],
#         duration=[duration],
#         total_stops=[total_stops],
#         additional_info=[additional_info],
#     )
# ).astype({col: "str" for col in ["date_of_journey", "dep_time", "arrival_time"]})

# if st.button("Predict"):
#     saved_preprocessor = joblib.load(r"models\preprocessor.pkl")
#     x_new_pre = saved_preprocessor.transform(x_new)

#     with open(r"models\final_xgboost_model.pkl", "rb") as f:
#         model = joblib.load(f)

#     pred = model.predict(x_new_pre)[0]

#     st.info(f"The predicted price is {pred:,.0f} INR")


import joblib
import numpy as np
import pandas as pd
import streamlit as st

# 1. Page Config
st.set_page_config(
    page_title="Flight Price Predictor",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# 2. Load Models (Cached for performance)
@st.cache_resource
def load_model_objects():
    try:
        # Update paths as necessary
        preprocessor = joblib.load(r"models/preprocessor.pkl")
        model = joblib.load(r"models/final_xgboost_model.pkl")
        return preprocessor, model
    except FileNotFoundError:
        st.error(
            "⚠️ Model files not found. Please ensure 'preprocessor.pkl' and 'final_xgboost_model.pkl' are in the 'models' directory."
        )
        return None, None


# 3. Custom CSS for UI Polish
st.markdown(
    """
    <style>
    .stButton>button {
        width: 100%;
        background-color: #FF4B4B;
        color: white;
        font-weight: bold;
        padding: 0.5rem;
    }
    .main_header {
        font-size: 2.5rem;
        color: #333;
        text-align: center;
        font-weight: 700;
    }
    .sub_header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    </style>
""",
    unsafe_allow_html=True,
)

# 4. Header Section
st.markdown(
    '<div class="main_header">✈️ Flight Price Predictor</div>', unsafe_allow_html=True
)
st.markdown(
    '<div class="sub_header">Powered by AWS SageMaker & XGBoost</div>',
    unsafe_allow_html=True,
)
st.markdown("---")

# 5. Main Layout
# We create a container to hold the inputs
with st.container():
    st.subheader("📋 Flight Details")

    # Row 1: Airline & Date
    col1, col2 = st.columns(2)
    with col1:
        airline = st.selectbox(
            "Airline",
            options=[
                "Jet Airways",
                "Indigo",
                "Air India",
                "Multiple Carriers",
                "Spicejet",
                "Vistara",
                "Air Asia",
                "Goair",
            ],
            help="Select the airline carrier",
        )
    with col2:
        doj = st.date_input("Date of Journey", help="Date of departure")

    # Row 2: Source & Destination
    col3, col4 = st.columns(2)
    with col3:
        source = st.selectbox(
            "Source Airport",
            options=["Mumbai", "Delhi", "Kolkata", "Banglore", "Chennai"],
        )
    with col4:
        destination = st.selectbox(
            "Destination Airport",
            options=[
                "Hyderabad",
                "Cochin",
                "Banglore",
                "Delhi",
                "New Delhi",
                "Kolkata",
            ],
        )

    # Validation: Source cannot be Destination
    if source == destination:
        st.warning("⚠️ Source and Destination cannot be the same city.")

    # Row 3: Timing Details (Grouped in an expander to save space)
    with st.expander("⏱️ Timing & Duration Details", expanded=True):
        c1, c2, c3 = st.columns(3)
        with c1:
            dep_time = st.time_input("Departure Time")
        with c2:
            arrival_time = st.time_input("Arrival Time")
        with c3:
            duration = st.number_input(
                "Duration (minutes)",
                step=5,
                min_value=30,
                help="Total flight duration in minutes",
            )

    # Row 4: Logistics
    c4, c5 = st.columns(2)
    with c4:
        total_stops = st.number_input("Total Stops", step=1, min_value=0, max_value=5)
    with c5:
        additional_info = st.selectbox(
            "Additional Info",
            options=[
                "No Info",
                "In-Flight Meal Not Included",
                "No Check-In Baggage Included",
                "1 Long Layover",
                "Change Airports",
                "Business Class",
                "Red-Eye Flight",
            ],
        )

# 6. Prediction Logic
st.markdown("---")
if st.button("Predict Flight Price"):
    if source == destination:
        st.error("Cannot predict: Source and Destination are the same.")
    else:
        # Prepare Data
        x_new = pd.DataFrame(
            {
                "airline": [airline],
                "date_of_journey": [doj],
                "source": [source],
                "destination": [destination],
                "dep_time": [dep_time],
                "arrival_time": [arrival_time],
                "duration": [duration],
                "total_stops": [total_stops],
                "additional_info": [additional_info],
            }
        )

        # Convert date/time objects to strings as expected by the preprocessor
        x_new = x_new.astype(
            {"date_of_journey": "str", "dep_time": "str", "arrival_time": "str"}
        )

        # Load models
        preprocessor, model = load_model_objects()

        if preprocessor and model:
            with st.spinner("Calculating best prices..."):
                try:
                    # Transform and Predict
                    x_new_pre = preprocessor.transform(x_new)
                    pred = model.predict(x_new_pre)[0]

                    # Display Result
                    st.success("Prediction Complete!")

                    # Create a centered column for the metric
                    m1, m2, m3 = st.columns([1, 2, 1])
                    with m2:
                        st.metric(
                            label="Estimated Ticket Price",
                            value=f"₹ {pred:,.0f}",
                            delta="INR",
                        )
                except Exception as e:
                    st.error(f"An error occurred during prediction: {e}")
