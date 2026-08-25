"""Streamlit frontend for Customer Churn Prediction.

This frontend calls a FastAPI backend for model inference.
Provides visualization and interactive forms for churn predictions.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st
import pandas as pd
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px

from app.core.config import get_settings
from app.frontend.services.analytics_service import AnalyticsDataProvider, AnalyticsSource
from app.frontend.services.api_client import ApiClient, ApiClientError
from app.frontend.services.model_comparison_service import (
    ComparisonReportError,
    ModelComparisonDataProvider,
)
from app.frontend.services.feature_expansion_service import (
    FeatureExpansionDataProvider,
    FeatureExpansionReportError,
)

# Configure page
st.set_page_config(
    page_title="Churn Prediction Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

SETTINGS = get_settings()
API_CLIENT = ApiClient(
    SETTINGS.api_url,
    timeout=SETTINGS.http_timeout,
    batch_timeout=SETTINGS.batch_http_timeout,
)

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


@st.cache_data(ttl=15)
def check_api_health():
    """Check if API is healthy."""
    try:
        return API_CLIENT.health_check().get("status") == "healthy"
    except ApiClientError:
        return False


@st.cache_data(ttl=60)
def get_model_info():
    """Get model information from API."""
    try:
        return API_CLIENT.get_model_info()
    except ApiClientError as e:
        st.error(f"Error fetching model info: {e}")
        return None


def make_prediction(customer_data):
    """Make a prediction via API."""
    try:
        return API_CLIENT.predict(customer_data)
    except ApiClientError as e:
        st.error(f"Error making prediction: {e}")
        return None


def make_batch_predictions(customers_list):
    """Make batch predictions via API."""
    try:
        return API_CLIENT.predict_batch(customers_list)
    except ApiClientError as e:
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
            ["Single Prediction", "Batch Predictions", "Model Info", "Analytics", "V2 vs V3 Comparison", "V3 vs V4 Features"]
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
    elif page == "V2 vs V3 Comparison":
        show_model_comparison()
    elif page == "V3 vs V4 Features":
        show_feature_expansion()


def show_single_prediction():
    """Single customer prediction page."""
    st.header("Bank Customer Churn Prediction")
    
    st.info("Enter bank customer details. V2/v3 use the three core fields; v4 uses the expanded bank profile.")
    
    # Create form in columns
    col1, col2 = st.columns(2)
    
    with col1:
        credit_score = st.slider("Credit Score", 300, 900, 650)
        age = st.slider("Age", 18, 100, 45)
        num_of_products = st.selectbox("Number of Products", [1, 2, 3, 4], index=1)
        tenure = st.slider("Tenure (years)", 0, 10, 5)
        balance = st.number_input("Balance", 0.0, 300000.0, 100000.0)
        estimated_salary = st.number_input("Estimated Salary", 0.0, 300000.0, 75000.0)
    
    with col2:
        geography = st.selectbox("Geography", ["France", "Germany", "Spain"])
        gender = st.selectbox("Gender", ["Female", "Male"])
        has_credit_card = st.selectbox("Has Credit Card", [1, 0], format_func=lambda value: "Yes" if value else "No")
        is_active = st.selectbox("Active Member", [1, 0], format_func=lambda value: "Yes" if value else "No")
        satisfaction = st.slider("Satisfaction Score", 1, 5, 3)
        card_type = st.selectbox("Card Type", ["DIAMOND", "GOLD", "SILVER", "PLATINUM"])
        points = st.number_input("Points Earned", 0, 2000, 500)
    
    # Prediction button
    if st.button("🔮 Predict Churn", use_container_width=True, type="primary"):
        customer_data = {
            "CreditScore": credit_score, "Geography": geography, "Gender": gender,
            "Age": age, "Tenure": tenure, "Balance": balance,
            "NumOfProducts": num_of_products, "HasCrCard": has_credit_card,
            "IsActiveMember": is_active, "EstimatedSalary": estimated_salary,
            "SatisfactionScore": satisfaction, "CardType": card_type,
            "PointEarned": points,
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
    
    provider = AnalyticsDataProvider()
    st.warning("DEMO DATA — synthetic examples, not real customer or prediction history.")
    st.caption("Analytics source: {}".format(provider.source.value))
    assert provider.source == AnalyticsSource.DEMO
    df = provider.get_data()
    
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


def show_model_comparison():
    """Reproduce the persisted v3 evaluation beside the production v2 baseline."""
    st.header("V2 vs V3 Model Comparison")
    st.caption("Fixed holdout results persisted by the promotion pipeline; no model is retrained on this page.")

    try:
        report = ModelComparisonDataProvider().load()
    except ComparisonReportError as exc:
        st.error("Comparison reports are unavailable: {}".format(exc))
        return

    decision = report.decision
    champion = decision["metrics"]["champion"]
    challenger = decision["metrics"]["challenger"]
    st.info(
        "Production remains on v2. Technical decision: {}. Automatic promotion: {}.".format(
            decision.get("status", "UNKNOWN"), "approved" if decision.get("approved") else "not approved"
        )
    )

    st.subheader("Headline results")
    metric_specs = [
        ("PR-AUC", "pr_auc", True),
        ("ROC-AUC", "roc_auc", True),
        ("Recall", "recall", True),
        ("F1", "f1", True),
        ("Brier score", "brier_score", False),
        ("Relative cost", "relative_cost", False),
    ]
    columns = st.columns(3)
    for index, (label, key, higher_is_better) in enumerate(metric_specs):
        v2_value = float(champion[key])
        v3_value = float(challenger[key])
        delta = v3_value - v2_value
        delta_color = "normal" if higher_is_better else "inverse"
        with columns[index % 3]:
            st.metric(
                "{} — v3".format(label),
                "{:.4f}".format(v3_value) if key != "relative_cost" else "{:.0f}".format(v3_value),
                "{:+.4f} vs v2".format(delta) if key != "relative_cost" else "{:+.0f} vs v2".format(delta),
                delta_color=delta_color,
            )

    st.subheader("Classification quality")
    quality = report.scenarios.loc[
        report.scenarios["scenario"].isin(["v2_current", "v3_balanced"]),
        ["scenario", "precision", "recall", "f1", "roc_auc", "pr_auc"],
    ].melt(id_vars="scenario", var_name="metric", value_name="score")
    fig = px.bar(quality, x="metric", y="score", color="scenario", barmode="group", range_y=[0, 1])
    st.plotly_chart(fig, use_container_width=True)

    left, right = st.columns(2)
    with left:
        st.subheader("Confusion matrices")
        matrices = pd.DataFrame({
            "Outcome": ["True negative", "False positive", "False negative", "True positive"],
            "v2 current": [champion["tn"], champion["fp"], champion["fn"], champion["tp"]],
            "v3 balanced": [challenger["tn"], challenger["fp"], challenger["fn"], challenger["tp"]],
        })
        st.dataframe(matrices, use_container_width=True, hide_index=True)
    with right:
        st.subheader("Operational impact")
        uplift = decision.get("uplift", {})
        st.metric("Additional churners identified", uplift.get("additional_churners_identified", 0))
        st.metric("False negatives avoided", uplift.get("false_negatives_avoided", 0))
        st.metric("Additional customers contacted", uplift.get("additional_customers_contacted", 0))
        st.metric("Campaign volume change", "{:+.1f}%".format(uplift.get("campaign_volume_change_pct", 0)))

    st.subheader("V3 threshold profiles")
    profiles = report.scenarios.loc[
        report.scenarios["model_version"] == "v3",
        ["scenario", "threshold", "precision", "recall", "f1", "campaign_rate", "relative_cost"],
    ].copy()
    st.dataframe(
        profiles.style.format({
            "threshold": "{:.2f}", "precision": "{:.1%}", "recall": "{:.1%}",
            "f1": "{:.3f}", "campaign_rate": "{:.1%}", "relative_cost": "{:.0f}",
        }),
        use_container_width=True,
        hide_index=True,
    )

    sensitivity = report.cost_sensitivity.copy()
    sensitivity["cost_ratio"] = sensitivity["fn_fp_ratio"].map(lambda value: "{:.0f}:1".format(value))
    fig = px.line(
        sensitivity,
        x="cost_ratio",
        y="best_threshold",
        markers=True,
        title="Best v3 threshold by false-negative / false-positive cost ratio",
        hover_data=["precision", "recall", "predicted_positive_rate", "cost"],
    )
    st.plotly_chart(fig, use_container_width=True)

    agreement = decision.get("agreement", {})
    st.subheader("Champion/challenger agreement")
    a1, a2, a3 = st.columns(3)
    a1.metric("Agreement rate", "{:.1%}".format(agreement.get("agreement_rate", 0)))
    a2.metric("Mean probability delta", "{:+.4f}".format(agreement.get("mean_probability_delta", 0)))
    a3.metric("P95 absolute delta", "{:.4f}".format(agreement.get("p95_probability_delta", 0)))

    if decision.get("warnings"):
        st.warning("Pending validation: " + " | ".join(decision["warnings"]))


def show_feature_expansion():
    """Present the leakage-safe feature ablation and v3/v4 holdout comparison."""
    st.header("V3 vs V4 — Bank Feature Expansion")
    st.caption("Persisted experiment results. V2 remains the production champion; this page does not retrain or promote models.")
    try:
        report = FeatureExpansionDataProvider().load()
    except FeatureExpansionReportError as exc:
        st.error(str(exc))
        return

    experiment = report.experiment
    st.info("Decision: {} | Selected group: {}".format(
        experiment.get("decision", "UNKNOWN"), experiment.get("selected_group", "UNKNOWN")
    ))
    readiness = experiment.get("readiness", {})
    if readiness:
        status = readiness.get("status", "UNKNOWN")
        (st.success if status == "READY_FOR_CANARY" else st.warning)(
            "Readiness: {} | Blocking gates: {}".format(
                status, ", ".join(readiness.get("blocking_gates", [])) or "none",
            )
        )
        st.dataframe(pd.DataFrame(readiness.get("gates", [])), use_container_width=True, hide_index=True)
    st.subheader("Selected leakage-safe features")
    st.write(", ".join(experiment.get("selected_features", [])))
    st.caption("Explicitly excluded: " + ", ".join(experiment.get("excluded_features", [])))

    st.subheader("Incremental feature ablation (training CV only)")
    ablation = report.ablation.sort_values("cv_pr_auc_mean", ascending=False)
    fig = px.bar(
        ablation, x="feature_group", y="cv_pr_auc_mean", color="feature_count",
        error_y="cv_pr_auc_std", title="Mean 5-fold PR-AUC by feature group",
        labels={"cv_pr_auc_mean": "CV PR-AUC", "feature_group": "Feature group"},
    )
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(
        ablation[["feature_group", "feature_count", "cv_pr_auc_mean", "cv_pr_auc_std", "cv_roc_auc_mean"]]
        .style.format({"cv_pr_auc_mean": "{:.4f}", "cv_pr_auc_std": "{:.4f}", "cv_roc_auc_mean": "{:.4f}"}),
        use_container_width=True, hide_index=True,
    )

    st.subheader("Untouched holdout: v3 versus v4")
    metrics = report.holdout.melt(
        id_vars=["model"], value_vars=["precision", "recall", "f1", "roc_auc", "pr_auc"],
        var_name="metric", value_name="score",
    )
    st.plotly_chart(px.bar(metrics, x="metric", y="score", color="model", barmode="group", range_y=[0, 1]), use_container_width=True)

    rows = report.holdout.set_index("model")
    v3, v4 = rows.loc["v3"], rows.loc["v4"]
    columns = st.columns(4)
    for column, (label, key, inverse) in zip(columns, [
        ("PR-AUC", "pr_auc", False), ("Recall", "recall", False),
        ("F1", "f1", False), ("Brier score", "brier_score", True),
    ]):
        with column:
            delta = float(v4[key] - v3[key])
            st.metric(label + " — v4", "{:.4f}".format(v4[key]), "{:+.4f} vs v3".format(delta),
                      delta_color="inverse" if inverse else "normal")

    st.subheader("V4 operating point")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Threshold", "{:.2f}".format(v4["threshold"]))
    c2.metric("True positives", int(v4["tp"]), int(v4["tp"] - v3["tp"]))
    c3.metric("False negatives", int(v4["fn"]), int(v4["fn"] - v3["fn"]), delta_color="inverse")
    c4.metric("False positives", int(v4["fp"]), int(v4["fp"] - v3["fp"]), delta_color="inverse")

    if "relative_cost" in rows.columns:
        st.subheader("Operational comparison (relative FN:FP cost = 5:1)")
        o1, o2, o3 = st.columns(3)
        o1.metric("V4 campaign rate", "{:.1%}".format(v4["campaign_rate"]),
                  "{:+.1%} vs v3".format(v4["campaign_rate"] - v3["campaign_rate"]))
        o2.metric("V4 relative cost", "{:.0f}".format(v4["relative_cost"]),
                  "{:+.0f} vs v3".format(v4["relative_cost"] - v3["relative_cost"]), delta_color="inverse")
        o3.metric("Churn coverage", "{:.1%}".format(v4["churn_coverage"]),
                  "{:+.1%} vs v3".format(v4["churn_coverage"] - v3["churn_coverage"]))

    st.subheader("Paired bootstrap — 95% confidence intervals")
    st.dataframe(report.bootstrap.style.format({
        "mean_delta": "{:+.4f}", "ci_95_lower": "{:+.4f}",
        "ci_95_upper": "{:+.4f}", "improvement_probability": "{:.1%}",
    }), use_container_width=True, hide_index=True)

    st.subheader("Fairness gaps")
    st.dataframe(report.fairness_gaps.style.format({
        "recall_gap": "{:.1%}", "false_positive_rate_gap": "{:.1%}",
        "brier_gap": "{:.4f}", "selection_rate_gap": "{:.1%}",
    }), use_container_width=True, hide_index=True)

    st.subheader("Permutation feature importance")
    st.plotly_chart(px.bar(
        report.feature_importance, x="importance_mean", y="feature", orientation="h",
        error_x="importance_std", title="PR-AUC decrease after feature permutation",
    ), use_container_width=True)


if __name__ == "__main__":
    main()
