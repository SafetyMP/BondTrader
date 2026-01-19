"""
Streamlit Dashboard for 2025 Bond Predictions
Displays predictions made by trained models on 2025 bond data
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Load .env file if it exists
try:
    from dotenv import load_dotenv

    env_path = Path(__file__).parent.parent / ".env"
    if env_path.exists():
        load_dotenv(env_path)
except ImportError:
    pass

from bondtrader.config import get_config
from bondtrader.utils import format_currency as _format_currency
from bondtrader.utils import format_percentage as _format_percentage

# Page configuration
st.set_page_config(
    page_title="2025 Bond Predictions Dashboard",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .stMetric {
        background-color: white;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def load_predictions(file_path: str) -> Optional[pd.DataFrame]:
    """Load predictions from CSV file"""
    if not os.path.exists(file_path):
        return None

    df = pd.read_csv(file_path)
    # Convert date strings to datetime
    if "maturity_date" in df.columns:
        df["maturity_date"] = pd.to_datetime(df["maturity_date"])

    return df


def format_currency(value: float) -> str:
    """Format number as currency with NaN handling"""
    if pd.isna(value):
        return "N/A"
    return _format_currency(value)


def format_percentage(value: float) -> str:
    """Format number as percentage with NaN handling"""
    if pd.isna(value):
        return "N/A"
    return _format_percentage(value)


def get_prediction_columns(df: pd.DataFrame) -> List[str]:
    """Extract all prediction column names from dataframe"""
    return [col for col in df.columns if "_predicted_value" in col]


def create_prediction_comparison_chart(df: pd.DataFrame) -> Optional[go.Figure]:
    """Create chart comparing predictions from different models"""
    prediction_cols = get_prediction_columns(df)

    if not prediction_cols:
        return None

    # Create comparison data
    comparison_data = []
    for _, row in df.iterrows():
        theoretical_val = row.get("theoretical_fair_value")
        current_price_val = row.get("current_price")

        if theoretical_val is not None:
            base_value = theoretical_val
        elif current_price_val is not None:
            base_value = current_price_val
        else:
            base_value = 1000

        for col in prediction_cols:
            if pd.notna(row[col]):
                model_name = col.replace("_predicted_value", "").replace("_", " ").title()
                predicted_val = row[col]
                difference = predicted_val - base_value

                if base_value > 0:
                    diff_percentage = (difference / base_value) * 100
                else:
                    diff_percentage = 0

                comparison_data.append(
                    {
                        "Bond ID": row["bond_id"],
                        "Model": model_name,
                        "Predicted Value": predicted_val,
                        "Theoretical Value": base_value,
                        "Difference": difference,
                        "Difference %": diff_percentage,
                    }
                )

    if not comparison_data:
        return None

    comp_df = pd.DataFrame(comparison_data)

    # Create grouped bar chart
    fig = px.bar(
        comp_df,
        x="Bond ID",
        y="Predicted Value",
        color="Model",
        barmode="group",
        title="Model Predictions Comparison",
        labels={"Predicted Value": "Predicted Value ($)", "Bond ID": "Bond ID"},
    )
    fig.update_layout(height=500, xaxis_tickangle=-45)
    return fig


def create_price_distribution_chart(df: pd.DataFrame) -> Optional[go.Figure]:
    """Create distribution chart of predicted prices"""
    prediction_cols = get_prediction_columns(df)

    if not prediction_cols:
        return None

    fig = go.Figure()

    for col in prediction_cols:
        if col in df.columns and df[col].notna().any():
            model_name = col.replace("_predicted_value", "").replace("_", " ").title()
            fig.add_trace(go.Histogram(x=df[col].dropna(), name=model_name, opacity=0.7, nbinsx=30))

    fig.update_layout(
        title="Distribution of Predicted Values",
        xaxis_title="Predicted Value ($)",
        yaxis_title="Frequency",
        barmode="overlay",
        height=400,
    )

    return fig


def calculate_model_performance_metrics(df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    """Calculate and return model performance metrics"""
    prediction_cols = get_prediction_columns(df)

    if not prediction_cols:
        return {}

    metrics = {}
    theoretical_col = "theoretical_fair_value"

    if theoretical_col not in df.columns:
        return metrics

    for col in prediction_cols:
        if col not in df.columns:
            continue

        model_name = col.replace("_predicted_value", "").replace("_", " ").title()
        pred_values = df[col].dropna()
        theo_values = df[theoretical_col].loc[pred_values.index]

        if len(pred_values) == 0:
            continue

        # Calculate metrics
        mae = (pred_values - theo_values).abs().mean()
        mse = ((pred_values - theo_values) ** 2).mean()
        rmse = mse**0.5
        mape = ((pred_values - theo_values).abs() / theo_values).mean() * 100

        # Correlation
        correlation = pred_values.corr(theo_values) if len(pred_values) > 1 else 0

        metrics[model_name] = {
            "MAE": mae,
            "RMSE": rmse,
            "MAPE": mape,
            "Correlation": correlation,
            "Count": len(pred_values),
        }

    return metrics


def apply_filters(df: pd.DataFrame) -> pd.DataFrame:
    """Apply sidebar filters to dataframe"""
    filtered_df = df.copy()

    # Bond type filter
    if "bond_type" in filtered_df.columns:
        bond_types = ["All"] + sorted(filtered_df["bond_type"].unique().tolist())
        selected_type = st.sidebar.selectbox("Bond Type", bond_types)
        if selected_type != "All":
            filtered_df = filtered_df[filtered_df["bond_type"] == selected_type]

    # Credit rating filter
    if "credit_rating" in filtered_df.columns:
        ratings = ["All"] + sorted(filtered_df["credit_rating"].unique().tolist())
        selected_rating = st.sidebar.selectbox("Credit Rating", ratings)
        if selected_rating != "All":
            filtered_df = filtered_df[filtered_df["credit_rating"] == selected_rating]

    # Issuer filter
    if "issuer" in filtered_df.columns:
        issuers = ["All"] + sorted(filtered_df["issuer"].unique().tolist())
        selected_issuer = st.sidebar.selectbox("Issuer", issuers)
        if selected_issuer != "All":
            filtered_df = filtered_df[filtered_df["issuer"] == selected_issuer]

    return filtered_df


def display_summary_statistics(df: pd.DataFrame):
    """Display summary statistics metrics"""
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Bonds", len(df))

    with col2:
        if "current_price" in df.columns:
            avg_price = df["current_price"].mean()
            st.metric("Average Current Price", format_currency(avg_price))

    with col3:
        if "theoretical_fair_value" in df.columns:
            avg_fair = df["theoretical_fair_value"].mean()
            st.metric("Average Fair Value", format_currency(avg_fair))

    with col4:
        if "coupon_rate" in df.columns:
            avg_coupon = df["coupon_rate"].mean()
            st.metric("Average Coupon Rate", format_percentage(avg_coupon))


def display_model_performance_metrics(metrics: Dict[str, Dict[str, float]]):
    """Display model performance metrics and visualizations"""
    if not metrics:
        st.info("No performance metrics available")
        return

    metrics_df = pd.DataFrame(metrics).T
    metrics_df = metrics_df.round(2)
    st.dataframe(metrics_df, use_container_width=True)

    # Visualize metrics
    metric_fig = go.Figure()

    for metric_name in ["MAE", "RMSE", "MAPE"]:
        if metric_name in metrics_df.columns:
            metric_fig.add_trace(go.Bar(x=metrics_df.index, y=metrics_df[metric_name], name=metric_name))

    metric_fig.update_layout(
        title="Model Performance Comparison",
        xaxis_title="Model",
        yaxis_title="Metric Value",
        barmode="group",
        height=400,
    )
    st.plotly_chart(metric_fig, use_container_width=True)


def display_detailed_predictions_table(df: pd.DataFrame):
    """Display detailed predictions table with relevant columns"""
    display_cols = [
        "bond_id",
        "bond_type",
        "issuer",
        "credit_rating",
        "coupon_rate",
        "current_price",
        "theoretical_fair_value",
    ]

    # Add prediction columns
    prediction_cols = get_prediction_columns(df)
    display_cols.extend(prediction_cols)

    # Filter to available columns
    display_cols = [col for col in display_cols if col in df.columns]

    st.dataframe(df[display_cols], use_container_width=True, height=400)


def main():
    """Main dashboard function"""
    config = get_config()

    # Header
    st.markdown('<h1 class="main-header">2025 Bond Predictions Dashboard</h1>', unsafe_allow_html=True)

    # Load predictions
    predictions_path = os.path.join(config.data_dir, "predictions", "2025_predictions.csv")

    if not os.path.exists(predictions_path):
        st.error(f"Predictions file not found at: {predictions_path}")
        st.info("Please run the training script first:")
        st.code("python scripts/train_evaluate_with_api_data.py")
        return

    df = load_predictions(predictions_path)

    if df is None or df.empty:
        st.error("No predictions data available")
        return

    # Sidebar filters
    st.sidebar.header("Filters")
    df = apply_filters(df)

    # Display summary statistics
    st.header("Summary Statistics")
    display_summary_statistics(df)

    # Model Performance Metrics
    st.header("Model Performance Metrics")
    metrics = calculate_model_performance_metrics(df)
    display_model_performance_metrics(metrics)

    # Predictions Comparison
    st.header("Predictions Comparison")
    comparison_fig = create_prediction_comparison_chart(df)
    if comparison_fig:
        st.plotly_chart(comparison_fig, use_container_width=True)
    else:
        st.info("No prediction data available for comparison")

    # Price Distribution
    st.header("Price Distribution")
    dist_fig = create_price_distribution_chart(df)
    if dist_fig:
        st.plotly_chart(dist_fig, use_container_width=True)
    else:
        st.info("No prediction data available for distribution chart")

    # Detailed Predictions Table
    st.header("Detailed Predictions")
    display_detailed_predictions_table(df)

    # Download button
    csv = df.to_csv(index=False)
    st.download_button(
        label="Download Predictions as CSV",
        data=csv,
        file_name="2025_predictions.csv",
        mime="text/csv",
    )

    # Model Information
    st.sidebar.header("Model Information")
    st.sidebar.info(
        """
        **Models Used:**
        - Basic ML Adjuster
        - Enhanced ML Adjuster
        - Advanced ML Adjuster (Ensemble)
        - AutoML

        **Training Data:** 2016-2017
        **Fine-tuning Data:** 2018
        **Prediction Data:** 2025
        """
    )


if __name__ == "__main__":
    main()
