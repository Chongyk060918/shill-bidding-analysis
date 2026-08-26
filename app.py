import streamlit as st
import pandas as pd
import joblib

# 1. Import your custom transformer
from custom_transforms import BoxplotWinsorizer

# 2. THE FIX: Tell Python's unpickler where to find the class
import __main__
__main__.BoxplotWinsorizer = BoxplotWinsorizer

# --- 1. PAGE SETUP ---
st.set_page_config(page_title="Auction Shield | Fraud Detection", page_icon="🛡️", layout="wide")

# --- 2. LOAD MODELS (Cached for speed) ---
@st.cache_resource
def load_models():
    models = {
        "Logistic Regression (Baseline)": joblib.load('logistic_regression_model.pkl'),
        "Random Forest Classifier": joblib.load('random_forest_model.pkl'),
        "Hist Gradient Boosting": joblib.load('hist_gradient_boosting_model.pkl')
    }
    return models

models = load_models()

# ... (The rest of your code remains exactly the same below this!)

# --- 3. HEADER ---
st.title("🛡️ Auction Shield: Shill Bidding Detection")
st.markdown("Real-time financial security and risk analytics portal for online auction integrity.")
st.divider()

# --- 4. TABS NAVIGATION ---
tab1, tab2, tab3 = st.tabs(["🔍 Live Bidder Inspection", "📁 Batch Auction Audit", "📊 Model Hub"])

# ==========================================
# MODULE 1: LIVE BIDDER INSPECTION (TAB 1)
# ==========================================
with tab1:
    st.subheader("Interactive Bidder Simulator")
    st.markdown("Adjust the behavioral metrics below to test the machine learning models in real-time.")
    
    # Model Selector
    selected_model_name = st.selectbox("Select Fraud Detection Algorithm:", list(models.keys()))
    active_model = models[selected_model_name]
    
    st.divider()
    
    # Input Sliders (Organized in columns for a clean UI)
    col1, col2, col3 = st.columns(3)
    
    with col1:
        bidder_tendency = st.slider("Bidder Tendency", 0.0, 1.0, 0.5)
        bidding_ratio = st.slider("Bidding Ratio", 0.0, 1.0, 0.2)
        winning_ratio = st.slider("Winning Ratio", 0.0, 1.0, 0.0)
        
    with col2:
        successive_outbidding = st.slider("Successive Outbidding", 0.0, 1.0, 0.0)
        last_bidding = st.slider("Last Bidding", 0.0, 1.0, 0.0)
        early_bidding = st.slider("Early Bidding", 0.0, 1.0, 0.0)
        
    with col3:
        auction_bids = st.slider("Auction Bids", 0.0, 10.0, 1.0)
        starting_price_average = st.slider("Starting Price Average", 0.0, 1.0, 0.0)
        auction_duration = st.slider("Auction Duration (Days)", 1, 10, 7)

    # Predict Button
    if st.button("Analyze Bidder Risk", type="primary", use_container_width=True):
        # Package inputs into a DataFrame matching training data
        input_data = pd.DataFrame([[
            bidder_tendency, bidding_ratio, winning_ratio, successive_outbidding,
            last_bidding, auction_bids, starting_price_average, early_bidding, auction_duration
        ]], columns=[
            "Bidder_Tendency", "Bidding_Ratio", "Winning_Ratio", "Successive_Outbidding",
            "Last_Bidding", "Auction_Bids", "Starting_Price_Average", "Early_Bidding", "Auction_Duration"
        ])
        
        # Make Prediction
        prediction = active_model.predict(input_data)[0]
        probability = active_model.predict_proba(input_data)[0][1]
        
        # Display Results
        st.divider()
        if prediction == 1:
            st.error(f"🚨 **FRAUDULENT BIDDER DETECTED (SHILL)**")
            st.warning(f"**Risk Confidence Score:** {probability * 100:.2f}% probability of shill activity.")
        else:
            st.success(f"✅ **LEGITIMATE BIDDER (NORMAL)**")
            st.info(f"**Risk Confidence Score:** {probability * 100:.2f}% probability of shill activity.")

# ==========================================
# MODULE 2 & 3: PLACEHOLDERS
# ==========================================
with tab2:
    st.info("Batch Audit Module will be built here next.")

with tab3:
    st.info("Model Performance Hub will be built here next.")
