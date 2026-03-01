import streamlit as st
import pickle
import numpy as np

# -------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------
st.set_page_config(
    page_title="Queue Waiting Time Prediction System",
    page_icon="⏳",
    layout="centered"
)

# -------------------------------------------------
# LOAD MODEL FILES
# -------------------------------------------------
@st.cache_resource
def load_files():
    model = pickle.load(open("model/waiting_time_model.pkl", "rb"))
    scaler = pickle.load(open("model/scaler.pkl", "rb"))
    encoder = pickle.load(open("model/queue_encoder.pkl", "rb"))
    return model, scaler, encoder

model, scaler, encoder = load_files()

# -------------------------------------------------
# TITLE & INTRODUCTION
# -------------------------------------------------
st.title("⏳ Queue Waiting Time Prediction System")

st.markdown(
    """
    **Welcome! 👋**

    This website helps you **estimate how long you may have to wait in a queue**  
    (for example: hospital, bank, ticket counter, supermarket, etc.).

    👉 You only need to enter **simple, real-life details**.  
    👉 No technical knowledge is required.
    """
)

st.divider()

# -------------------------------------------------
# INPUT SECTION
# -------------------------------------------------
st.header("📝 Enter Queue Information")

st.markdown(
    "Please answer the following questions based on the **current situation of the queue**."
)

# -------------------------------
# Number of People
# -------------------------------
num_people = st.number_input(
    "👥 How many people are currently standing in the queue?",
    min_value=1,
    max_value=500,
    value=50
)
st.caption("Example: If you see around 50 people waiting, enter 50.")

# -------------------------------
# Counters Open
# -------------------------------
counters = st.number_input(
    "🏧 How many service counters are open?",
    min_value=1,
    max_value=20,
    value=5
)
st.caption("Example: If 5 counters/windows are serving people, enter 5.")

# -------------------------------
# Average Service Time
# -------------------------------
avg_service_time = st.number_input(
    "⏱️ On average, how many minutes does it take to serve ONE person?",
    min_value=1,
    max_value=60,
    value=10
)
st.caption("Example: If one person takes about 10 minutes, enter 10.")

# -------------------------------
# Time of Day (USER FRIENDLY)
# -------------------------------
time_of_day = st.slider(
    "🕰️ What is the current time of day?",
    min_value=0,
    max_value=23,
    value=12
)

st.caption(
    """
    **How to select time:**  
    - 9 = 9 AM  
    - 12 = 12 Noon  
    - 15 = 3 PM  
    - 18 = 6 PM  
    - 21 = 9 PM  
    """
)

# -------------------------------
# Queue Type (WORDS ONLY)
# -------------------------------
queue_type = st.selectbox(
    "🏥 What type of place is this queue?",
    list(encoder.classes_)
)

st.caption(
    "Example: Hospital, Bank, Ticket Counter, Supermarket, etc."
)

st.divider()

# -------------------------------------------------
# PREDICTION SECTION
# -------------------------------------------------
st.header("📊 Estimated Waiting Time")

if st.button("🔍 Predict Waiting Time", use_container_width=True):

    # Convert user-friendly input to model format
    queue_encoded = encoder.transform([queue_type])[0]

    input_data = np.array([[
        num_people,
        counters,
        avg_service_time,
        time_of_day,
        queue_encoded
    ]])

    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0]

    st.success(
        f"🕒 **You may need to wait approximately {prediction:.2f} minutes**"
    )

    st.markdown(
        """
        ℹ️ **Important Note:**  
        This prediction is based on historical data and machine learning analysis.  
        Actual waiting time may change depending on real-time conditions.
        """
    )

# -------------------------------------------------
# FOOTER
# -------------------------------------------------
st.divider()
st.caption(
    "Queue Waiting Time Prediction System | Machine Learning Based Web Application"
)