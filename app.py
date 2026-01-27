"""
Customer Churn Prediction Dashboard
====================================
Interactive Streamlit dashboard for predicting customer churn using the trained Gradient Boosting model.

This dashboard allows users to:
- Input customer data
- Get real-time churn predictions
- View model performance metrics
- Analyze feature importance
- Monitor prediction confidence

Author: Data Science Team
Date: 2026-01-27
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from model_deployment import ModelDeployment
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set page configuration
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
    <style>
    .main {
        padding-top: 0rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
    }
    .churn-risk-high {
        color: #d62728;
        font-weight: bold;
    }
    .churn-risk-low {
        color: #2ca02c;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)

# Initialize session state
if 'deployment' not in st.session_state:
    st.session_state.deployment = ModelDeployment()
    st.session_state.deployment.load_all()

if 'prediction_history' not in st.session_state:
    st.session_state.prediction_history = []


def load_model():
    """Load model artifacts on first run."""
    deployment = st.session_state.deployment
    if not deployment.is_loaded:
        st.error("⚠️ Model artifacts not loaded. Please ensure models directory exists with saved model files.")
        return False
    return True


def create_prediction_input():
    """
    Create input form for customer data.
    
    Returns:
        dict: Dictionary with user input values
    """
    st.sidebar.header("📋 Customer Information")
    
    # Organize inputs in expandable sections
    with st.sidebar.expander("💼 Product & Service Info", expanded=True):
        num_products = st.slider(
            "Number of Products",
            min_value=1,
            max_value=4,
            value=2,
            help="Number of products the customer has with the bank"
        )
    
    with st.sidebar.expander("📅 Tenure Information", expanded=True):
        age = st.slider(
            "Age (years)",
            min_value=18,
            max_value=100,
            value=35,
            help="Customer's age in years"
        )
        tenure = st.slider(
            "Tenure (months)",
            min_value=0,
            max_value=10,
            value=5,
            help="Customer tenure with the bank in months"
        )
    
    # Prepare input data
    input_data = {
        'NumOfProducts': num_products,
        'Age_Squared_StandardScaled': (age ** 2),  # Will be scaled by model
        'Age_Tenure_Interaction_MinMaxScaled': (age * tenure)  # Will be scaled by model
    }
    
    return input_data


def make_prediction(input_data):
    """
    Make prediction using the loaded model.
    
    Args:
        input_data (dict): Dictionary with input features
    
    Returns:
        dict: Prediction results
    """
    deployment = st.session_state.deployment
    
    # Preprocess input
    processed_data = deployment.preprocess_input(input_data)
    
    if processed_data is None:
        st.error("❌ Error preprocessing input data")
        return None
    
    # Make prediction
    result = deployment.predict(processed_data)
    return result


def display_prediction_result(result, input_data):
    """
    Display prediction results in a visually appealing format.
    
    Args:
        result (dict): Prediction result dictionary
        input_data (dict): Input data used for prediction
    """
    col1, col2, col3 = st.columns(3)
    
    # Churn prediction
    with col1:
        st.metric(
            "Prediction",
            result['prediction_label'],
            delta=f"Confidence: {result['confidence']:.1%}",
            delta_color="off"
        )
    
    # Probability of churn
    with col2:
        prob_churn = result['probability_churned']
        st.metric(
            "Churn Probability",
            f"{prob_churn:.1%}",
            delta=f"Risk Level: {'🔴 High' if prob_churn > 0.5 else '🟡 Medium' if prob_churn > 0.25 else '🟢 Low'}",
            delta_color="off"
        )
    
    # Probability of retention
    with col3:
        prob_retained = result['probability_retained']
        st.metric(
            "Retention Probability",
            f"{prob_retained:.1%}",
            delta=f"Loyalty: {'🟢 Strong' if prob_retained > 0.75 else '🟡 Moderate' if prob_retained > 0.5 else '🔴 Weak'}",
            delta_color="off"
        )
    
    # Visualization of prediction probabilities
    st.write("---")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Probability gauge
        fig, ax = plt.subplots(figsize=(10, 3))
        categories = ['Retained', 'Churned']
        probabilities = [result['probability_retained'], result['probability_churned']]
        colors = ['#2ca02c', '#d62728']
        
        bars = ax.barh(categories, probabilities, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
        ax.set_xlabel('Probability', fontsize=12, fontweight='bold')
        ax.set_title('Prediction Probability Distribution', fontsize=14, fontweight='bold')
        ax.set_xlim([0, 1])
        
        # Add probability labels
        for i, (bar, prob) in enumerate(zip(bars, probabilities)):
            ax.text(prob + 0.02, i, f'{prob:.1%}', va='center', fontsize=11, fontweight='bold')
        
        plt.tight_layout()
        st.pyplot(fig)
    
    with col2:
        # Risk level indicator
        st.write("")
        st.write("")
        if result['prediction'] == 1:
            st.error(f"🔴 **HIGH RISK**\nCustomer is predicted to churn")
        else:
            st.success(f"🟢 **LOW RISK**\nCustomer is likely to stay")


def display_customer_summary(input_data):
    """
    Display summary of input customer data.
    
    Args:
        input_data (dict): Dictionary with input features
    """
    st.write("---")
    st.subheader("📊 Customer Profile Summary")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.info(f"**Number of Products:** {input_data['NumOfProducts']}")
    
    with col2:
        # Reverse engineer age from Age_Squared
        age = np.sqrt(abs(input_data['Age_Squared_StandardScaled']))
        st.info(f"**Age (approx):** {age:.0f} years")
    
    with col3:
        # Calculate tenure from interaction
        age = np.sqrt(abs(input_data['Age_Squared_StandardScaled']))
        tenure = input_data['Age_Tenure_Interaction_MinMaxScaled'] / (age + 1) if age > 0 else 0
        st.info(f"**Tenure (approx):** {tenure:.0f} months")


def display_model_info():
    """Display information about the model."""
    st.sidebar.write("---")
    st.sidebar.subheader("📈 Model Information")
    
    st.sidebar.info(
        """
        **Model:** Gradient Boosting Classifier
        
        **Performance:**
        - Accuracy: 83.85%
        - Precision: 65.23%
        - Recall: 44.61%
        - F1-Score: 52.98%
        - ROC-AUC: 82.66%
        
        **Features Used:** 3
        1. NumOfProducts
        2. Age_Squared_StandardScaled
        3. Age_Tenure_Interaction_MinMaxScaled
        
        **Training Data:** 8,000 samples
        **Test Data:** 2,000 samples
        """
    )


def main():
    """Main function to run the Streamlit dashboard."""
    
    # Header
    st.title("🎯 Customer Churn Prediction Dashboard")
    st.markdown("""
        ### Predict customer churn risk using machine learning
        Enter customer information to get real-time churn predictions powered by Gradient Boosting
    """)
    
    # Load model
    if not load_model():
        st.stop()
    
    # Create tabs for different sections
    tab1, tab2, tab3 = st.tabs(["🔮 Prediction", "📊 Analytics", "ℹ️ About"])
    
    with tab1:
        st.header("Make a Prediction")
        st.markdown("Adjust the customer parameters in the sidebar and get instant churn predictions")
        
        # Create input form
        input_data = create_prediction_input()
        
        # Prediction button
        if st.button("🚀 Predict Churn Risk", use_container_width=True):
            with st.spinner("Analyzing customer data..."):
                result = make_prediction(input_data)
            
            if result:
                st.success("✅ Prediction completed!")
                display_prediction_result(result, input_data)
                display_customer_summary(input_data)
                
                # Store in history
                st.session_state.prediction_history.append({
                    'input': input_data,
                    'result': result
                })
            else:
                st.error("❌ Error making prediction")
    
    with tab2:
        st.header("📊 Model Performance & Analytics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Model Metrics")
            metrics_data = {
                'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC'],
                'Score': [0.8385, 0.6523, 0.4461, 0.5298, 0.8266]
            }
            metrics_df = pd.DataFrame(metrics_data)
            
            fig, ax = plt.subplots(figsize=(8, 5))
            bars = ax.barh(metrics_df['Metric'], metrics_df['Score'], color='steelblue', alpha=0.7, edgecolor='black')
            ax.set_xlabel('Score', fontsize=11, fontweight='bold')
            ax.set_title('Gradient Boosting Model Performance', fontsize=12, fontweight='bold')
            ax.set_xlim([0, 1])
            
            for bar, score in zip(bars, metrics_df['Score']):
                ax.text(score + 0.02, bar.get_y() + bar.get_height()/2, 
                       f'{score:.1%}', va='center', fontsize=10, fontweight='bold')
            
            plt.tight_layout()
            st.pyplot(fig)
        
        with col2:
            st.subheader("Model Comparison")
            comparison_data = {
                'Model': ['Logistic Regression', 'Random Forest', 'Gradient Boosting', 'Ensemble (GB+RF)'],
                'F1-Score': [0.0442, 0.4740, 0.5298, 0.5066]
            }
            comparison_df = pd.DataFrame(comparison_data)
            
            fig, ax = plt.subplots(figsize=(8, 5))
            colors = ['#ff6b6b', '#4ecdc4', '#45b7d1', '#ffd93d']
            bars = ax.bar(range(len(comparison_df)), comparison_df['F1-Score'], color=colors, 
                         alpha=0.7, edgecolor='black', linewidth=2)
            ax.set_xticks(range(len(comparison_df)))
            ax.set_xticklabels(comparison_df['Model'], rotation=45, ha='right')
            ax.set_ylabel('F1-Score', fontsize=11, fontweight='bold')
            ax.set_title('Model Comparison (F1-Score)', fontsize=12, fontweight='bold')
            ax.set_ylim([0, 1])
            
            for bar, score in zip(bars, comparison_df['F1-Score']):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                       f'{score:.1%}', ha='center', va='bottom', fontsize=10, fontweight='bold')
            
            plt.tight_layout()
            st.pyplot(fig)
    
    with tab3:
        st.header("ℹ️ About This Dashboard")
        
        st.subheader("📌 Overview")
        st.markdown("""
            This dashboard uses a **Gradient Boosting machine learning model** to predict customer churn risk.
            The model was trained on 10,000 customer records with an 83.85% accuracy rate.
        """)
        
        st.subheader("🎯 How It Works")
        st.markdown("""
            1. **Input**: Enter customer information (number of products, age, tenure)
            2. **Processing**: The model analyzes the input features
            3. **Prediction**: Get a churn probability and risk classification
            4. **Insight**: Understand the likelihood of customer churn
        """)
        
        st.subheader("📊 Model Details")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Algorithm", "Gradient Boosting")
        
        with col2:
            st.metric("Training Samples", "8,000")
        
        with col3:
            st.metric("Features Used", "3")
        
        st.subheader("📈 Dataset Information")
        st.markdown("""
            - **Total Records:** 10,000 customers
            - **Churn Rate:** 20.38%
            - **Train/Test Split:** 80/20
            - **Feature Engineering:** 12 engineered features → 5 final features (after multicollinearity removal)
        """)
        
        st.subheader("💡 Key Features Used")
        st.markdown("""
            1. **NumOfProducts**: Number of products customer has
            2. **Age_Squared_StandardScaled**: Polynomial feature capturing non-linear age effects
            3. **Age_Tenure_Interaction_MinMaxScaled**: Interaction between age and tenure
        """)
        
        st.subheader("⚠️ Disclaimer")
        st.warning("""
            This model provides predictions based on historical data. 
            Actual churn may vary. Use alongside business judgment and domain expertise.
        """)
    
    # Display model info in sidebar
    display_model_info()
    
    # Show prediction history if available
    if st.session_state.prediction_history:
        st.sidebar.write("---")
        st.sidebar.subheader("📜 Recent Predictions")
        
        for i, pred in enumerate(st.session_state.prediction_history[-5:], 1):
            st.sidebar.write(f"{i}. {pred['result']['prediction_label']} ({pred['result']['confidence']:.1%})")


if __name__ == "__main__":
    main()
