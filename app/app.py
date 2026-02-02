"""Streamlit frontend for Customer Churn Prediction.

This frontend calls a FastAPI backend for model inference.
Provides visualization and interactive forms for churn predictions.
"""

import streamlit as st
import requests
import pandas as pd
import numpy as np
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px

# Configure page
st.set_page_config(
    page_title="Churn Prediction Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# API Configuration
API_URL = "http://localhost:8000" #st.secrets.get("API_URL", "http://localhost:8000")

# Styling
st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 8px;
        text-align: center;
    }
    .prediction-high {
        background-color: #fff3cd;
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid #ffc107;
    }
    .prediction-low {
        background-color: #d4edda;
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid #28a745;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def check_api_health():
    """Check if API is healthy."""
    try:
        response = requests.get(f"{API_URL}/health", timeout=5)
        return response.status_code == 200
    except:
        return False


@st.cache_data
def get_model_info():
    """Get model information from API."""
    try:
        response = requests.get(f"{API_URL}/model-info", timeout=5)
        if response.status_code == 200:
            return response.json()
        else:
            return None
    except Exception as e:
        st.error(f"Error fetching model info: {e}")
        return None


def make_prediction(customer_data):
    """Make a prediction via API."""
    try:
        response = requests.post(
            f"{API_URL}/predict",
            json=customer_data,
            timeout=10
        )
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"API Error: {response.status_code}")
            return None
    except Exception as e:
        st.error(f"Error making prediction: {e}")
        return None


def make_batch_predictions(customers_list):
    """Make batch predictions via API."""
    try:
        response = requests.post(
            f"{API_URL}/predict-batch",
            json={"data": customers_list},
            timeout=30
        )
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"API Error: {response.status_code}")
            return None
    except Exception as e:
        st.error(f"Error making batch predictions: {e}")
        return None


# Main app
def main():
    st.title("📊 Customer Churn Prediction Dashboard")
    
    # Check API health
    if not check_api_health():
        st.error("⚠️ API is not running. Please start the API server with: `python -m uvicorn app.api:app --reload`")
        return
    
    st.success("✅ API Connected")
    
    # Sidebar
    with st.sidebar:
        st.header("Navigation")
        page = st.radio(
            "Select Page",
            ["Single Prediction", "Batch Predictions", "Model Info", "Analytics"]
        )
        
        st.divider()
        
        # Model info in sidebar
        model_info = get_model_info()
        if model_info:
            st.subheader("Model Information")
            st.metric("Model Type", model_info.get('model_type', 'N/A'))
            st.metric("Version", model_info.get('version', 'N/A'))
            st.metric("Features", model_info.get('n_features', 0))
    
    # Pages
    if page == "Single Prediction":
        show_single_prediction()
    elif page == "Batch Predictions":
        show_batch_predictions()
    elif page == "Model Info":
        show_model_info()
    elif page == "Analytics":
        show_analytics()


def show_single_prediction():
    """Single customer prediction page."""
    st.header("Single Customer Prediction")
    
    st.info("Enter customer details to predict churn probability.")
    
    # Create form in columns
    col1, col2 = st.columns(2)
    
    with col1:
        senior_citizen = st.selectbox("Senior Citizen", [0, 1], format_func=lambda x: "Yes" if x else "No")
        tenure = st.slider("Tenure (months)", 0, 72, 24)
        monthly_charges = st.number_input("Monthly Charges ($)", 0.0, 200.0, 65.5)
        total_charges = st.number_input("Total Charges ($)", 0.0, 10000.0, 1570.0)
    
    with col2:
        internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
        online_security = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
        online_backup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
        device_protection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
    
    col3, col4 = st.columns(2)
    
    with col3:
        tech_support = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
        streaming_tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
        streaming_movies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
    
    with col4:
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        payment_method = st.selectbox(
            "Payment Method",
            ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]
        )
    
    # Prediction button
    if st.button("🔮 Predict Churn", use_container_width=True, type="primary"):
        customer_data = {
            "SeniorCitizen": senior_citizen,
            "Tenure": tenure,
            "MonthlyCharges": monthly_charges,
            "TotalCharges": total_charges,
            "InternetService": internet_service,
            "OnlineSecurity": online_security,
            "OnlineBackup": online_backup,
            "DeviceProtection": device_protection,
            "TechSupport": tech_support,
            "StreamingTV": streaming_tv,
            "StreamingMovies": streaming_movies,
            "Contract": contract,
            "PaymentMethod": payment_method
        }
        
        with st.spinner("Making prediction..."):
            result = make_prediction(customer_data)
        
        if result:
            prediction = result['prediction']
            probability = result['probability']
            confidence = result['confidence']
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if prediction == 1:
                    st.metric("Prediction", "🔴 CHURN", "High Risk")
                else:
                    st.metric("Prediction", "🟢 STAY", "Low Risk")
            
            with col2:
                st.metric("Churn Probability", f"{probability:.1%}")
            
            with col3:
                st.metric("Confidence", confidence.upper())
            
            # Gauge chart
            fig = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=probability * 100,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Churn Probability (%)"},
                delta={'reference': 50},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "red" if probability > 0.5 else "green"},
                    'steps': [
                        {'range': [0, 25], 'color': "#d4edda"},
                        {'range': [25, 50], 'color': "#fff3cd"},
                        {'range': [50, 75], 'color': "#f8d7da"},
                        {'range': [75, 100], 'color': "#f5c6cb"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 50
                    }
                }
            ))
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
            
            # Recommendation
            st.divider()
            if prediction == 1:
                st.warning("⚠️ **Recommendation:** High churn risk detected. Consider proactive retention strategies.")
            else:
                st.success("✅ **Recommendation:** Low churn risk. Continue monitoring.")


def show_batch_predictions():
    """Batch predictions page."""
    st.header("Batch Predictions")
    
    st.info("Upload a CSV file with customer data to make batch predictions.")
    
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        
        st.subheader("Data Preview")
        st.dataframe(df.head())
        
        st.metric("Total Rows", len(df))
        st.metric("Total Columns", len(df.columns))
        
        if st.button("🔮 Predict All", use_container_width=True, type="primary"):
            # Convert to list of dicts
            customers_list = df.to_dict('records')
            
            with st.spinner(f"Making predictions for {len(customers_list)} customers..."):
                result = make_batch_predictions(customers_list)
            
            if result:
                predictions_list = result['predictions']
                
                # Create results dataframe
                results_df = pd.DataFrame([
                    {
                        'Prediction': 'Churn' if p['prediction'] == 1 else 'Stay',
                        'Probability': p['probability'],
                        'Confidence': p['confidence']
                    }
                    for p in predictions_list
                ])
                
                # Combine with original data
                final_df = pd.concat([df.reset_index(drop=True), results_df], axis=1)
                
                st.subheader("Predictions")
                st.dataframe(final_df, use_container_width=True)
                
                # Statistics
                col1, col2, col3 = st.columns(3)
                with col1:
                    churn_count = (results_df['Prediction'] == 'Churn').sum()
                    st.metric("Predicted Churn", churn_count, f"{churn_count/len(results_df)*100:.1f}%")
                with col2:
                    avg_prob = results_df['Probability'].mean()
                    st.metric("Avg Churn Probability", f"{avg_prob:.1%}")
                with col3:
                    high_confidence = (results_df['Confidence'] == 'high').sum()
                    st.metric("High Confidence", high_confidence)
                
                # Download results
                csv = final_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Predictions (CSV)",
                    data=csv,
                    file_name=f"predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )


def show_model_info():
    """Model information page."""
    st.header("Model Information")
    
    model_info = get_model_info()
    
    if model_info:
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Model Type", model_info.get('model_type', 'N/A'))
            st.metric("Version", model_info.get('version', 'N/A'))
            st.metric("Training Score", f"{model_info.get('accuracy', 0):.2%}")
        
        with col2:
            st.metric("Total Features", model_info.get('n_features', 0))
            st.metric("F1 Score", f"{model_info.get('f1_score', 0) or 'N/A'}")
        
        st.subheader("Features Used")
        features = model_info.get('feature_names', [])
        
        # Create columns for features
        cols = st.columns(3)
        for idx, feature in enumerate(features):
            with cols[idx % 3]:
                st.write(f"• {feature}")


def show_analytics():
    """Analytics page (demo)."""
    st.header("Analytics Dashboard")
    
    st.info("This page demonstrates analytics capabilities with sample data.")
    
    # Sample data for demo
    np.random.seed(42)
    n_samples = 100
    
    df = pd.DataFrame({
        'Tenure': np.random.randint(0, 72, n_samples),
        'MonthlyCharges': np.random.uniform(20, 120, n_samples),
        'TotalCharges': np.random.uniform(0, 8000, n_samples),
        'Prediction': np.random.choice(['Stay', 'Churn'], n_samples, p=[0.8, 0.2]),
        'Probability': np.random.uniform(0, 1, n_samples)
    })
    
    # Metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Customers", len(df))
    with col2:
        churn_pct = (df['Prediction'] == 'Churn').sum() / len(df) * 100
        st.metric("Predicted Churn %", f"{churn_pct:.1f}%")
    with col3:
        st.metric("Avg Monthly Charges", f"${df['MonthlyCharges'].mean():.2f}")
    with col4:
        st.metric("Avg Tenure (months)", f"{df['Tenure'].mean():.0f}")
    
    st.divider()
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.pie(
            df,
            names='Prediction',
            title='Churn Distribution',
            color_discrete_map={'Stay': '#28a745', 'Churn': '#dc3545'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.histogram(
            df,
            x='Probability',
            nbins=20,
            title='Churn Probability Distribution',
            color_discrete_sequence=['#0066cc']
        )
        st.plotly_chart(fig, use_container_width=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.scatter(
            df,
            x='Tenure',
            y='MonthlyCharges',
            color='Prediction',
            title='Tenure vs Monthly Charges',
            color_discrete_map={'Stay': '#28a745', 'Churn': '#dc3545'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.box(
            df,
            x='Prediction',
            y='Probability',
            title='Churn Probability by Prediction',
            color='Prediction',
            color_discrete_map={'Stay': '#28a745', 'Churn': '#dc3545'}
        )
        st.plotly_chart(fig, use_container_width=True)


if __name__ == "__main__":
    main()