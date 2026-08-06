import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier # Imported to train the model on the fly
import matplotlib.pyplot as plt

# --- PAGE SETUP ---
st.set_page_config(page_title="Shill Bidding Detector", page_icon="🕵️", layout="wide")

# --- LOAD DATA ---
@st.cache_data
def load_data():
    return pd.read_csv('Shill Bidding Dataset.csv')

df = load_data()

# --- TRAIN MODEL ON THE FLY (NO JOBLIB NEEDED) ---
@st.cache_resource
def train_fraud_model(data):
    # 1. Isolate the features (drop IDs and target)
    # We drop IDs because they are just labels, not predictive math features
    X = data.drop(columns=['Record_ID', 'Auction_ID', 'Bidder_ID', 'Class'])
    # 2. Isolate the target (What we want to predict)
    y = data['Class']
    
    # 3. Train a fast Random Forest model
    model = RandomForestClassifier(random_state=42, n_estimators=50)
    model.fit(X, y)
    
    # Return the trained model and the column names it used
    return model, X.columns

rf_model, feature_cols = train_fraud_model(df)

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
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🚨 Live Fraud Monitoring", 
    "🕵️ Bidder Behavior Profiling", 
    "🕸️ Auction Anomalies",
    "📊 Confidence Scores",
    "new",
    "table"
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
        
# TAB 4: Confidence Scores (NEW IMPLEMENTATION)
with tab4:
    st.header("Fraud Confidence Scores")
    st.write("Select a specific bid record to evaluate the model's statistical confidence.")
    
    # Create a dropdown for the user to select a record ID from the filtered data
    record_to_test = st.selectbox(
        "Select a Record_ID to analyze:", 
        filtered_df['Record_ID'].head(100) # Limiting to 100 for a cleaner dropdown
    )

    if record_to_test:
        # 1. Fetch the exact row of data the user selected
        record_data = filtered_df[filtered_df['Record_ID'] == record_to_test]
        
        # 2. Extract only the math features the model was trained on
        X_test = record_data[feature_cols]
        
        # 3. Get the probabilities: Outputs [Probability of Normal, Probability of Fraud]
        probabilities = rf_model.predict_proba(X_test)[0]
        
        # 4. Format for Plotly
        prob_df = pd.DataFrame({
            "Status": ["Normal Bid (Class 0)", "Fraudulent Bid (Class 1)"],
            "Probability": probabilities
        })
        
        st.subheader(f"AI Analysis for Record: {record_to_test}")
        
        # 5. Display a beautiful bar chart using Plotly
        fig4 = px.bar(
            prob_df, 
            x="Status", 
            y="Probability", 
            color="Status", 
            range_y=[0, 1], # Lock Y-axis from 0 to 1 (0% to 100%)
            color_discrete_map={"Normal Bid (Class 0)": "green", "Fraudulent Bid (Class 1)": "red"}
        )
        st.plotly_chart(fig4, use_container_width=True)
with tab5:
    counts = df['Class'].value_counts()

    labels = ['Non-Shill (0)', 'Shill (1)']
    colors = ['#1f77b4', '#ff7f0e']
    
    plt.figure(figsize=(7, 7))
    
    # 3. Create the pie chart
    plt.pie(
        counts.values,
        labels=labels,
        colors=colors,
        autopct='%1.1f%%',          # Display percentages with one decimal place
    )
    
    plt.title('Distribution of Classes (Shill vs Non-Shill)')
    st.pyplot()
with tab6:
    import pandas as pd

    data_frame = pd.read_csv('Shill Bidding Dataset.csv')
    print(data_frame.head())
