import streamlit as st
import pandas as pd
import plotly.express as px
import joblib

# --- PAGE SETUP ---
st.set_page_config(page_title="Shill Bidding Detector", page_icon="🕵️", layout="wide")

# --- LOAD DATA ---
@st.cache_data
def load_data():
    return pd.read_csv('Shill Bidding Dataset.csv')

df = load_data()

# 1. Load the model efficiently at the top of your script
@st.cache_resource
def load_model():
    # Replace with your actual saved model file
    return joblib.load("my_text_classifier.pkl")

pipeline = load_model()

st.title("20 Newsgroups Text Classifier")

# --- SIDEBAR ---
st.sidebar.title("⚙️ Control Panel")
st.sidebar.divider()
selected_duration = st.sidebar.slider(
    "Filter by Auction Duration (Days)", 
    int(df['Auction_Duration'].min()), 
    int(df['Auction_Duration'].max()), 
    (1, 10)
)

# Filter dataframe based on sidebar
filtered_df = df[(df['Auction_Duration'] >= selected_duration[0]) & 
                 (df['Auction_Duration'] <= selected_duration[1])]

# --- MAIN HEADER ---
st.title("Auction Fraud & Shill Bidding Detector")
st.markdown("**System Status:** Active Monitoring | **Dataset:** Shill Bidding Dataset.csv")
st.divider()



# --- TABS ---

tab1, tab2, tab3, tab4 = st.tabs([
    "🚨 Live Fraud Monitoring", 
    "🕵️ Bidder Behavior Profiling", 
    "🕸️ Auction Anomalies",
    "Confidence Scores"
])
# TAB 1: Live Monitoring
with tab1:
    st.subheader("High-Risk Alerts & Overview")
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Unique Auctions", filtered_df['Auction_ID'].nunique())
    col2.metric("Total Tracked Bidders", filtered_df['Bidder_ID'].nunique())
    col3.metric("Flagged Fraudulent Bids", filtered_df[filtered_df['Class'] == 1].shape[0])
    
    st.markdown("#### Suspicious Activity Ledger (Class = 1)")
    fraud_df = filtered_df[filtered_df['Class'] == 1]
    st.dataframe(fraud_df[['Record_ID', 'Auction_ID', 'Bidder_ID', 'Successive_Outbidding', 'Winning_Ratio']], use_container_width=True)

# TAB 2: Behavior Profiling
with tab2:
    st.subheader("Bidding vs. Winning Ratios")
    fig1 = px.scatter(
        filtered_df, 
        x="Bidding_Ratio", 
        y="Winning_Ratio", 
        color="Class",
        hover_data=['Bidder_ID'],
        title="Identifying Bait Bidders (High Bidding, Low Winning)"
    )
    st.plotly_chart(fig1, use_container_width=True)

# TAB 3: Auction Anomalies
with tab3:
    st.subheader("Auction Dynamics")
    colA, colB = st.columns(2)
    
    with colA:
        fig2 = px.scatter(filtered_df, x="Starting_Price_Average", y="Auction_Bids", color="Class", title="Starting Price vs Total Bids")
        st.plotly_chart(fig2, use_container_width=True)
        
    with colB:
        fig3 = px.scatter(filtered_df, x="Early_Bidding", y="Last_Bidding", color="Class", title="Bidding Timeline Anomalies")
        st.plotly_chart(fig3, use_container_width=True)
        
with tab4:
    st.header("Confidence Scores")
    st.write("Analyze the statistical likelihood of the top 5 categories.")
    
    # Note: We add a unique 'key' to prevent Streamlit widget duplication errors 
    # if you also have a text_area in Tab 1!
    user_input = st.text_area("Enter text to evaluate:", key="confidence_input")

    if user_input:
        # Extract classes and probabilities
        classes = pipeline.classes_
        probabilities = pipeline.predict_proba([user_input])[0]
        
        # Create a sorted Pandas DataFrame
        prob_df = pd.DataFrame({
            "Category": classes,
            "Probability": probabilities
        })
        
        # Isolate the top 5 most likely categories
        top_5_df = prob_df.sort_values(by="Probability", ascending=False).head(5)
        
        # Display the results
        st.subheader("Top 5 Predictions")
        st.bar_chart(data=top_5_df, x="Category", y="Probability")

