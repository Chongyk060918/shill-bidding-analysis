import streamlit as st
import pandas as pd
import joblib

# 1. Import your custom transformer exactly as named in your folder!
from custom_transforms import BoxplotWinsorizer

# 2. THE CLOUD FIX: Tell Python's unpickler where to find the class in memory
import __main__
__main__.BoxplotWinsorizer = BoxplotWinsorizer

# --- 1. PAGE SETUP ---
st.set_page_config(page_title="Auction Shield | Fraud Detection", page_icon="🛡️", layout="wide")

# --- 2. MODEL PATHS (LAZY LOADING) ---
# We ONLY store the file paths here. We do not load them into memory yet!
MODEL_PATHS = {
    "Logistic Regression (Baseline)": 'logistic_regression_model.pkl',
    "Random Forest Classifier": 'random_forest_model.pkl',
    "Hist Gradient Boosting": 'hist_gradient_boosting_model.pkl'
}

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
    selected_model_name = st.selectbox("Select Fraud Detection Algorithm:", list(MODEL_PATHS.keys()))
    
    st.divider()
    
    # Input Sliders
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
        
        # LAZY LOAD THE MODEL HERE: It only loads the one you picked, saving RAM!
        with st.spinner(f"Loading {selected_model_name} and analyzing..."):
            active_model = joblib.load(MODEL_PATHS[selected_model_name])
            
            # Package inputs into a DataFrame matching EXACT training data order
            input_data = pd.DataFrame([[
                bidder_tendency, 
                bidding_ratio, 
                successive_outbidding,
                last_bidding, 
                auction_bids, 
                starting_price_average, 
                early_bidding, 
                winning_ratio, 
                auction_duration
            ]], columns=[
                "Bidder_Tendency", "Bidding_Ratio", "Successive_Outbidding",
                "Last_Bidding", "Auction_Bids", "Starting_Price_Average", 
                "Early_Bidding", "Winning_Ratio", "Auction_Duration"
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
