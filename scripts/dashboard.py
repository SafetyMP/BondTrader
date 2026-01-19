"""
Streamlit Dashboard for Bond Trading Analysis
Interactive dashboard for bond valuation, comparison, and arbitrage detection
"""

from datetime import datetime, timedelta
from typing import Dict, List

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from bondtrader.analytics.alternative_data import AlternativeDataAnalyzer
from bondtrader.analytics.backtesting import BacktestEngine
from bondtrader.analytics.correlation_analysis import CorrelationAnalyzer
from bondtrader.analytics.execution_strategies import ExecutionStrategy
from bondtrader.analytics.factor_models import FactorModel
from bondtrader.analytics.floating_rate_bonds import FloatingRateBondPricer
from bondtrader.analytics.key_rate_duration import KeyRateDuration
from bondtrader.analytics.multi_curve import MultiCurveFramework
from bondtrader.analytics.oas_pricing import OASPricer
from bondtrader.analytics.portfolio_optimization import PortfolioOptimizer
from bondtrader.config import get_config
from bondtrader.core.arbitrage_detector import ArbitrageDetector
from bondtrader.core.bond_models import Bond, BondType
from bondtrader.core.bond_valuation import BondValuator
from bondtrader.data.data_generator import BondDataGenerator
from bondtrader.ml.automl import AutoMLBondAdjuster
from bondtrader.ml.bayesian_optimization import BayesianOptimizer
from bondtrader.ml.ml_adjuster import MLBondAdjuster
from bondtrader.ml.ml_advanced import AdvancedMLBondAdjuster
from bondtrader.ml.regime_models import RegimeDetector
from bondtrader.risk.credit_risk_enhanced import CreditRiskEnhanced
from bondtrader.risk.liquidity_risk_enhanced import LiquidityRiskEnhanced
from bondtrader.risk.risk_management import RiskManager
from bondtrader.risk.tail_risk import TailRiskAnalyzer
from bondtrader.utils.auth import get_user_manager, logout, require_auth
from bondtrader.utils.rate_limiter import get_dashboard_rate_limiter

# Page configuration
st.set_page_config(page_title="Bond Trading Dashboard", page_icon="📊", layout="wide", initial_sidebar_state="expanded")

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
    </style>
""",
    unsafe_allow_html=True,
)


@st.cache_data
def load_sample_bonds(num_bonds: int = 50) -> List[Bond]:
    """Load or generate sample bonds"""
    generator = BondDataGenerator()
    return generator.generate_bonds(num_bonds)


def format_currency(value: float) -> str:
    """Format number as currency"""
    return f"${value:,.2f}"


def format_percentage(value: float) -> str:
    """Format number as percentage"""
    return f"{value:.2f}%"


@require_auth
def main():
    """Main dashboard application"""
    # Get centralized configuration
    config = get_config()

    # Rate limiting check
    rate_limiter = get_dashboard_rate_limiter()
    username = st.session_state.get("username", "anonymous")
    allowed, error = rate_limiter.is_allowed(username)
    if not allowed:
        st.error(f"Rate limit exceeded: {error}")
        st.stop()

    # Logout button in sidebar
    with st.sidebar:
        if st.button("🚪 Logout"):
            logout()
            st.rerun()

        # Show user info
        if "username" in st.session_state:
            st.info(f"👤 Logged in as: **{st.session_state.username}**")

    st.markdown('<h1 class="main-header">📊 Bond Trading & Arbitrage Dashboard</h1>', unsafe_allow_html=True)

    # Sidebar controls
    st.sidebar.header("Configuration")

    # Use config default for risk-free rate, but allow override via slider
    default_rfr = config.default_risk_free_rate * 100
    risk_free_rate = st.sidebar.slider("Risk-Free Rate (%)", min_value=0.0, max_value=10.0, value=default_rfr, step=0.1) / 100

    num_bonds = st.sidebar.slider("Number of Bonds", min_value=10, max_value=200, value=50, step=10)

    # Use config default for min profit threshold
    default_min_profit = config.min_profit_threshold * 100
    min_profit_threshold = (
        st.sidebar.slider("Min Profit Threshold (%)", min_value=0.0, max_value=5.0, value=default_min_profit, step=0.1) / 100
    )

    use_ml = st.sidebar.checkbox("Use ML Adjustments", value=True)
    train_ml = st.sidebar.checkbox("Train ML Model", value=False)

    # Load bonds
    bonds = load_sample_bonds(num_bonds)

    # Initialize components
    valuator = BondValuator(risk_free_rate=risk_free_rate)
    # Use config for model type
    ml_adjuster = MLBondAdjuster(model_type=config.ml_model_type)

    # Train ML model if requested
    if train_ml and len(bonds) >= 10:
        with st.sidebar:
            with st.spinner("Training ML model..."):
                metrics = ml_adjuster.train(bonds, test_size=config.ml_test_size, random_state=config.ml_random_state)
                st.success("ML Model Trained!")
                st.metric("Train R²", f"{metrics['train_r2']:.3f}")
                st.metric("Test R²", f"{metrics['test_r2']:.3f}")
                st.metric("Test RMSE", f"{metrics['test_rmse']:.4f}")
    else:
        # Load or use default untrained model
        pass

    detector = ArbitrageDetector(
        valuator=valuator, ml_adjuster=ml_adjuster if use_ml else None, min_profit_threshold=min_profit_threshold
    )

    # Main tabs
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10, tab11, tab12 = st.tabs(
        [
            "📈 Overview",
            "💰 Arbitrage Opportunities",
            "🔍 Bond Comparison",
            "📊 Bond Details",
            "📉 Portfolio Analysis",
            "🎯 OAS & Options Analysis",
            "📏 Key Rate Duration",
            "⚠️ Risk Analytics",
            "⚖️ Portfolio Optimization",
            "🔬 Factor Models",
            "📈 Backtesting & Execution",
            "🚀 Advanced ML & AI",
        ]
    )

    # Tab 1: Overview
    with tab1:
        st.header("Market Overview")

        col1, col2, col3, col4 = st.columns(4)

        # Calculate summary statistics
        fair_values = [valuator.calculate_fair_value(b) for b in bonds]
        market_prices = [b.current_price for b in bonds]
        mismatches = [fv - mp for fv, mp in zip(fair_values, market_prices)]

        total_market_value = sum(market_prices)
        total_fair_value = sum(fair_values)
        total_mismatch = total_fair_value - total_market_value
        avg_mismatch_pct = np.mean([((fv - mp) / mp) * 100 for fv, mp in zip(fair_values, market_prices)])

        with col1:
            st.metric("Total Market Value", format_currency(total_market_value))
        with col2:
            st.metric("Total Fair Value", format_currency(total_fair_value))
        with col3:
            st.metric("Total Mismatch", format_currency(total_mismatch))
        with col4:
            color = "normal"
            if abs(avg_mismatch_pct) > 3:
                color = "inverse"
            st.metric("Avg Mismatch %", format_percentage(avg_mismatch_pct), delta=format_percentage(avg_mismatch_pct))

        # Bond type distribution
        st.subheader("Bond Distribution by Type")
        bond_type_counts = pd.Series([b.bond_type.value for b in bonds]).value_counts()

        col1, col2 = st.columns(2)
        with col1:
            fig_pie = px.pie(values=bond_type_counts.values, names=bond_type_counts.index, title="Bond Type Distribution")
            st.plotly_chart(fig_pie, use_container_width=True)

        with col2:
            # Price vs Fair Value scatter
            bond_data = []
            for bond, fv in zip(bonds, fair_values):
                bond_data.append(
                    {
                        "Bond ID": bond.bond_id,
                        "Type": bond.bond_type.value,
                        "Market Price": bond.current_price,
                        "Fair Value": fv,
                        "Mismatch %": ((fv - bond.current_price) / bond.current_price) * 100,
                    }
                )
            df_overview = pd.DataFrame(bond_data)

            fig_scatter = px.scatter(
                df_overview,
                x="Market Price",
                y="Fair Value",
                color="Type",
                hover_data=["Bond ID", "Mismatch %"],
                title="Market Price vs Fair Value",
                labels={"Market Price": "Market Price ($)", "Fair Value": "Fair Value ($)"},
            )
            fig_scatter.add_trace(
                go.Scatter(
                    x=[df_overview["Market Price"].min(), df_overview["Market Price"].max()],
                    y=[df_overview["Market Price"].min(), df_overview["Market Price"].max()],
                    mode="lines",
                    name="Fair Value Line",
                    line=dict(dash="dash", color="gray"),
                )
            )
            st.plotly_chart(fig_scatter, use_container_width=True)

    # Tab 2: Arbitrage Opportunities
    with tab2:
        st.header("Arbitrage Opportunities")

        opportunities = detector.find_arbitrage_opportunities(bonds, use_ml=use_ml)

        if opportunities:
            st.success(f"Found {len(opportunities)} arbitrage opportunities!")

            # Summary metrics
            col1, col2, col3 = st.columns(3)
            total_profit = sum([abs(o["profit_opportunity"]) for o in opportunities])
            avg_profit_pct = np.mean([abs(o["profit_percentage"]) for o in opportunities])
            buy_opportunities = sum([1 for o in opportunities if o["recommendation"] == "BUY"])

            with col1:
                st.metric("Total Profit Potential", format_currency(total_profit))
            with col2:
                st.metric("Average Profit %", format_percentage(avg_profit_pct))
            with col3:
                st.metric("Buy Opportunities", buy_opportunities)

            # Opportunities table
            opp_df = pd.DataFrame(opportunities)
            opp_df["market_price"] = opp_df["market_price"].apply(format_currency)
            opp_df["adjusted_fair_value"] = opp_df["adjusted_fair_value"].apply(format_currency)
            opp_df["profit_opportunity"] = opp_df["profit_opportunity"].apply(format_currency)
            opp_df["profit_percentage"] = opp_df["profit_percentage"].apply(format_percentage)
            opp_df["ytm"] = opp_df["ytm"].apply(format_percentage)

            st.dataframe(
                opp_df[
                    [
                        "bond_id",
                        "bond_type",
                        "issuer",
                        "recommendation",
                        "market_price",
                        "adjusted_fair_value",
                        "profit_percentage",
                        "ytm",
                        "credit_rating",
                        "arbitrage_type",
                    ]
                ].rename(
                    columns={
                        "bond_id": "Bond ID",
                        "bond_type": "Type",
                        "issuer": "Issuer",
                        "recommendation": "Action",
                        "market_price": "Market Price",
                        "adjusted_fair_value": "Fair Value",
                        "profit_percentage": "Profit %",
                        "ytm": "YTM",
                        "credit_rating": "Rating",
                        "arbitrage_type": "Opportunity Type",
                    }
                ),
                use_container_width=True,
                height=400,
            )

            # Profit visualization
            st.subheader("Profit Opportunity Analysis")
            fig_profit = px.bar(
                opp_df,
                x="bond_id",
                y="profit_percentage",
                color="recommendation",
                title="Profit Percentage by Bond",
                labels={"bond_id": "Bond ID", "profit_percentage": "Profit Percentage (%)"},
            )
            st.plotly_chart(fig_profit, use_container_width=True)

        else:
            st.info("No arbitrage opportunities found above the threshold. Try lowering the profit threshold.")

    # Tab 3: Bond Comparison
    with tab3:
        st.header("Bond Comparison")

        grouping = st.selectbox(
            "Group By",
            options=["bond_type", "credit_rating", "maturity_bucket"],
            format_func=lambda x: x.replace("_", " ").title(),
        )

        comparisons = detector.compare_equivalent_bonds(bonds, grouping_key=grouping)

        if comparisons:
            comp_df = pd.DataFrame(comparisons)

            # Group comparison chart
            fig_group = px.box(
                comp_df,
                x="group",
                y="relative_mispricing_pct",
                title=f"Mispricing Distribution by {grouping.replace('_', ' ').title()}",
                labels={"group": "Group", "relative_mispricing_pct": "Relative Mispricing (%)"},
            )
            st.plotly_chart(fig_group, use_container_width=True)

            # Detailed comparison table
            st.subheader("Detailed Comparison")
            display_df = comp_df.copy()
            display_df["market_price"] = display_df["market_price"].apply(format_currency)
            display_df["fair_value"] = display_df["fair_value"].apply(format_currency)
            display_df["relative_mispricing_pct"] = display_df["relative_mispricing_pct"].apply(format_percentage)

            st.dataframe(
                display_df[["bond_id", "group", "bond_type", "market_price", "fair_value", "relative_mispricing_pct"]].rename(
                    columns={
                        "bond_id": "Bond ID",
                        "group": "Group",
                        "bond_type": "Type",
                        "market_price": "Market Price",
                        "fair_value": "Fair Value",
                        "relative_mispricing_pct": "Mispricing %",
                    }
                ),
                use_container_width=True,
            )

    # Tab 4: Bond Details
    with tab4:
        st.header("Individual Bond Analysis")

        bond_ids = [b.bond_id for b in bonds]
        selected_id = st.selectbox("Select Bond", bond_ids)
        selected_bond = next(b for b in bonds if b.bond_id == selected_id)

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Bond Information")
            st.write(f"**Bond ID:** {selected_bond.bond_id}")
            st.write(f"**Type:** {selected_bond.bond_type.value}")
            st.write(f"**Issuer:** {selected_bond.issuer}")
            st.write(f"**Credit Rating:** {selected_bond.credit_rating}")
            st.write(f"**Face Value:** {format_currency(selected_bond.face_value)}")
            st.write(f"**Coupon Rate:** {format_percentage(selected_bond.coupon_rate)}")
            st.write(f"**Maturity Date:** {selected_bond.maturity_date.strftime('%Y-%m-%d')}")
            st.write(f"**Time to Maturity:** {selected_bond.time_to_maturity:.2f} years")

        with col2:
            st.subheader("Valuation Metrics")

            # Calculate metrics
            fv = valuator.calculate_fair_value(selected_bond)
            ytm = valuator.calculate_yield_to_maturity(selected_bond) * 100
            duration = valuator.calculate_duration(selected_bond)
            convexity = valuator.calculate_convexity(selected_bond)
            mismatch_data = valuator.calculate_price_mismatch(selected_bond)

            if use_ml and ml_adjuster.is_trained:
                ml_result = ml_adjuster.predict_adjusted_value(selected_bond)
                ml_fv = ml_result["ml_adjusted_fair_value"]
                adjustment = ml_result["adjustment_factor"]
            else:
                ml_fv = fv
                adjustment = 1.0

            st.metric("Market Price", format_currency(selected_bond.current_price))
            st.metric("Theoretical Fair Value", format_currency(fv))
            if use_ml and ml_adjuster.is_trained:
                st.metric("ML-Adjusted Fair Value", format_currency(ml_fv))
                st.metric("ML Adjustment Factor", f"{adjustment:.4f}")
            st.metric("Yield to Maturity", format_percentage(ytm))
            st.metric("Duration", f"{duration:.2f} years")
            st.metric("Convexity", f"{convexity:.2f}")
            st.metric(
                "Mismatch %",
                format_percentage(mismatch_data["mismatch_percentage"]),
                delta=format_percentage(mismatch_data["mismatch_percentage"]),
            )

        # Mismatch visualization
        st.subheader("Price Mismatch Analysis")
        fig_mismatch = go.Figure()

        x_labels = ["Market Price", "Theoretical Fair Value"]
        y_values = [selected_bond.current_price, fv]
        colors = ["#ff4444", "#44ff44"]

        if use_ml and ml_adjuster.is_trained:
            x_labels.append("ML-Adjusted Value")
            y_values.append(ml_fv)
            colors.append("#4444ff")

        fig_mismatch.add_trace(go.Bar(x=x_labels, y=y_values, name="Value", marker_color=colors))
        fig_mismatch.update_layout(title=f"Value Comparison for {selected_bond.bond_id}", yaxis_title="Value ($)")
        st.plotly_chart(fig_mismatch, use_container_width=True)

    # Tab 5: Portfolio Analysis
    with tab5:
        st.header("Portfolio Arbitrage Analysis")

        portfolio_analysis = detector.calculate_portfolio_arbitrage(bonds)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Portfolio Market Value", format_currency(portfolio_analysis["total_market_value"]))
        with col2:
            st.metric("Portfolio Fair Value", format_currency(portfolio_analysis["total_fair_value"]))
        with col3:
            profit_pct = portfolio_analysis["portfolio_profit_pct"]
            st.metric("Portfolio Profit %", format_percentage(profit_pct), delta=format_percentage(profit_pct))

        st.metric("Number of Opportunities", portfolio_analysis["num_opportunities"])
        st.metric("Average Opportunity %", format_percentage(portfolio_analysis["avg_opportunity_pct"]))

        # Portfolio composition
        st.subheader("Portfolio Composition")
        bond_types = [b.bond_type.value for b in bonds]
        type_counts = pd.Series(bond_types).value_counts()

        fig_composition = px.bar(
            x=type_counts.index, y=type_counts.values, title="Number of Bonds by Type", labels={"x": "Bond Type", "y": "Count"}
        )
        st.plotly_chart(fig_composition, use_container_width=True)

    # Tab 6: OAS & Options Analysis
    with tab6:
        st.header("Option-Adjusted Spread (OAS) Analysis")
        st.write("Industry-standard pricing for bonds with embedded options")

        callable_bonds = [b for b in bonds if b.callable]
        if callable_bonds:
            st.success(f"Found {len(callable_bonds)} callable bonds")

            oas_pricer = OASPricer(valuator)
            bond_ids = [b.bond_id for b in callable_bonds]
            selected_id = st.selectbox("Select Callable Bond", bond_ids, key="oas_bond")
            selected_bond = next(b for b in callable_bonds if b.bond_id == selected_id)

            volatility = st.slider("Volatility (%)", 0.0, 50.0, 15.0, step=0.5) / 100

            oas_result = oas_pricer.calculate_oas(selected_bond, volatility=volatility)

            if "error" not in oas_result:
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("OAS (bps)", f"{oas_result['oas_bps']:.2f}")
                    st.metric("Option Value", format_currency(oas_result["option_value"]))
                with col2:
                    st.metric("Market Price", format_currency(oas_result["market_price"]))
                    st.metric("Option-Free Value", format_currency(oas_result["option_free_value"]))
                with col3:
                    st.metric("OAS-Adjusted Value", format_currency(oas_result["option_adjusted_value"]))
                    st.metric("Volatility", f"{volatility*100:.1f}%")

                # OAS visualization
                st.subheader("OAS Analysis")
                fig_oas = go.Figure()
                fig_oas.add_trace(
                    go.Bar(
                        x=["Market Price", "Option-Free", "OAS-Adjusted"],
                        y=[oas_result["market_price"], oas_result["option_free_value"], oas_result["option_adjusted_value"]],
                        marker_color=["#ff4444", "#44ff44", "#4444ff"],
                    )
                )
                fig_oas.update_layout(title=f"OAS Analysis for {selected_bond.bond_id}", yaxis_title="Value ($)")
                st.plotly_chart(fig_oas, use_container_width=True)
            else:
                st.error(f"Error calculating OAS: {oas_result.get('error', 'Unknown error')}")
        else:
            st.info("No callable bonds found in portfolio")

    # Tab 7: Key Rate Duration
    with tab7:
        st.header("Key Rate Duration (KRD) Analysis")
        st.write("Sensitivity to specific points on the yield curve")

        krd_calculator = KeyRateDuration(valuator)

        # Individual bond KRD
        st.subheader("Individual Bond KRD")
        bond_ids = [b.bond_id for b in bonds]
        selected_id = st.selectbox("Select Bond", bond_ids, key="krd_bond")
        selected_bond = next(b for b in bonds if b.bond_id == selected_id)

        krd_result = krd_calculator.calculate_krd(selected_bond)

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Macaulay Duration", f"{krd_result['macaulay_duration']:.2f} years")
            st.metric("Total KRD", f"{krd_result['total_krd']:.2f} years")
        with col2:
            st.metric("Base Price", format_currency(krd_result["base_price"]))
            st.metric("Base YTM", format_percentage(krd_result["base_ytm"]))

        # KRD chart
        krd_df = pd.DataFrame({"Key Rate": [f"{r}y" for r in krd_result["key_rates"]], "KRD": krd_result["krd_values"]})

        fig_krd = px.bar(
            krd_df,
            x="Key Rate",
            y="KRD",
            title=f"Key Rate Duration for {selected_bond.bond_id}",
            labels={"KRD": "KRD (years)"},
        )
        st.plotly_chart(fig_krd, use_container_width=True)

        # Portfolio KRD
        st.subheader("Portfolio KRD Analysis")
        portfolio_krd = krd_calculator.calculate_portfolio_krd(bonds)

        portfolio_krd_df = pd.DataFrame(
            {"Key Rate": [f"{r}y" for r in portfolio_krd["key_rates"]], "Portfolio KRD": portfolio_krd["portfolio_krd_values"]}
        )

        fig_portfolio_krd = px.bar(
            portfolio_krd_df,
            x="Key Rate",
            y="Portfolio KRD",
            title="Portfolio Key Rate Duration",
            labels={"Portfolio KRD": "KRD (years)"},
        )
        st.plotly_chart(fig_portfolio_krd, use_container_width=True)

        # Yield curve shock scenarios
        st.subheader("Yield Curve Shock Analysis")
        shock_scenarios = st.multiselect(
            "Select Scenarios",
            options=["parallel_shift", "steepening", "flattening", "twist"],
            default=["parallel_shift", "steepening"],
        )

        if shock_scenarios:
            shock_result = krd_calculator.yield_curve_shock_analysis(bonds, shock_scenarios=shock_scenarios)

            scenario_data = []
            for scenario, data in shock_result["scenarios"].items():
                scenario_data.append(
                    {"Scenario": scenario.replace("_", " ").title(), "Portfolio Change %": data["portfolio_change_pct"]}
                )

            shock_df = pd.DataFrame(scenario_data)
            fig_shock = px.bar(
                shock_df,
                x="Scenario",
                y="Portfolio Change %",
                title="Portfolio Impact of Yield Curve Shocks",
                labels={"Portfolio Change %": "Change (%)"},
            )
            st.plotly_chart(fig_shock, use_container_width=True)

    # Tab 8: Risk Analytics
    with tab8:
        st.header("Comprehensive Risk Analytics")

        risk_subtab = st.radio("Select Risk Type", ["Credit Risk", "Liquidity Risk", "Multi-Curve Analysis"], horizontal=True)

        if risk_subtab == "Credit Risk":
            st.subheader("Enhanced Credit Risk Analysis")

            credit_risk = CreditRiskEnhanced(valuator)

            # Merton model analysis
            st.write("**Merton Structural Model**")
            bond_ids = [b.bond_id for b in bonds]
            selected_id = st.selectbox("Select Bond", bond_ids, key="credit_bond")
            selected_bond = next(b for b in bonds if b.bond_id == selected_id)

            asset_vol = st.slider("Asset Volatility (%)", 10.0, 50.0, 25.0, step=1.0) / 100

            merton_result = credit_risk.merton_structural_model(selected_bond, asset_volatility=asset_vol)

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Default Probability", f"{merton_result['default_probability']*100:.3f}%")
                st.metric("Distance to Default", f"{merton_result['distance_to_default']:.2f}")
            with col2:
                st.metric("Recovery Rate", f"{merton_result['recovery_rate']*100:.1f}%")
                st.metric("Expected Loss", format_currency(merton_result["expected_loss"]))
            with col3:
                st.metric("Loss Given Default", format_currency(merton_result["loss_given_default"]))

            # Credit migration analysis
            st.subheader("Credit Migration Analysis")
            migration_result = credit_risk.credit_migration_analysis(selected_bond)

            migration_df = pd.DataFrame(
                [
                    {
                        "Rating": rating,
                        "Probability": data["probability"] * 100,
                        "Expected Value": data["expected_value"],
                        "Value Change %": data["value_change_pct"],
                    }
                    for rating, data in migration_result["value_distribution"].items()
                ]
            )

            col1, col2 = st.columns(2)
            with col1:
                fig_migration_prob = px.bar(migration_df, x="Rating", y="Probability", title="Migration Probabilities")
                st.plotly_chart(fig_migration_prob, use_container_width=True)

            with col2:
                fig_migration_value = px.bar(migration_df, x="Rating", y="Value Change %", title="Value Impact by Rating")
                st.plotly_chart(fig_migration_value, use_container_width=True)

            # Credit VaR
            st.subheader("Credit Value at Risk (CVaR)")
            cvar_result = credit_risk.calculate_credit_var(bonds, confidence_level=0.95)

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Credit VaR", format_currency(cvar_result["credit_var"]))
            with col2:
                st.metric("Credit VaR %", format_percentage(cvar_result["credit_var_pct"]))
            with col3:
                st.metric("Portfolio Value", format_currency(cvar_result["current_portfolio_value"]))

        elif risk_subtab == "Liquidity Risk":
            st.subheader("Liquidity Risk Analysis")

            liquidity_risk = LiquidityRiskEnhanced(valuator)

            # Portfolio liquidity analysis
            liquidity_result = liquidity_risk.analyze_liquidity_risk(bonds)

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Spread Cost", format_currency(liquidity_result["total_spread_cost"]))
                st.metric("Avg Spread (bps)", f"{liquidity_result['avg_spread_bps']:.2f}")
            with col2:
                st.metric("Total Impact Cost", format_currency(liquidity_result["total_impact_cost"]))
                st.metric("Avg Depth Ratio", f"{liquidity_result['avg_depth_ratio']:.2f}")
            with col3:
                st.metric("Total Liquidity Cost", format_currency(liquidity_result["total_liquidity_cost"]))
                st.metric("Liquidity Cost %", format_percentage(liquidity_result["liquidity_cost_pct"]))

            # Liquidity-adjusted VaR
            st.subheader("Liquidity-Adjusted VaR (LVaR)")
            lvar_result = liquidity_risk.calculate_lvar(bonds, confidence_level=0.95)

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("LVaR", format_currency(lvar_result["lvar_value"]))
                st.metric("LVaR %", format_percentage(lvar_result["lvar_pct"]))
            with col2:
                st.metric("Standard VaR", format_currency(lvar_result["var_value"]))
                st.metric("Var %", format_percentage(lvar_result["var_pct"]))
            with col3:
                st.metric("Liquidity Adjustment", format_currency(lvar_result["liquidity_cost"]))
                st.metric("Adjustment %", format_percentage(lvar_result["liquidity_cost_pct"]))

            # Liquidity by bond
            liquidity_df = pd.DataFrame(liquidity_result["bond_metrics"])
            if not liquidity_df.empty:
                st.subheader("Liquidity Metrics by Bond")
                st.dataframe(
                    liquidity_df[["bond_id", "rating", "spread_bps", "depth_ratio", "liquidity_rating", "total_cost"]].rename(
                        columns={
                            "bond_id": "Bond ID",
                            "rating": "Rating",
                            "spread_bps": "Spread (bps)",
                            "depth_ratio": "Depth Ratio",
                            "liquidity_rating": "Liquidity Rating",
                            "total_cost": "Total Cost",
                        }
                    ),
                    use_container_width=True,
                )

        elif risk_subtab == "Multi-Curve Analysis":
            st.subheader("Multi-Curve Framework Analysis")
            st.write("Separate discounting (OIS) and forwarding (LIBOR/SOFR) curves")

            multi_curve = MultiCurveFramework(valuator)
            multi_curve.initialize_default_curves()

            # Show curves
            st.write("**Curve Comparison**")

            if multi_curve.ois_curve and multi_curve.libor_curve:
                curve_df = pd.DataFrame(
                    {
                        "Maturity": multi_curve.ois_curve["maturities"],
                        "OIS Rate": [r * 100 for r in multi_curve.ois_curve["rates"]],
                        "LIBOR Rate": [r * 100 for r in multi_curve.libor_curve["rates"]],
                        "Basis Spread": [
                            multi_curve.calculate_basis_spread(m) * 10000 for m in multi_curve.ois_curve["maturities"]
                        ],
                    }
                )

                fig_curves = go.Figure()
                fig_curves.add_trace(
                    go.Scatter(
                        x=curve_df["Maturity"],
                        y=curve_df["OIS Rate"],
                        mode="lines+markers",
                        name="OIS Curve",
                        line=dict(color="blue"),
                    )
                )
                fig_curves.add_trace(
                    go.Scatter(
                        x=curve_df["Maturity"],
                        y=curve_df["LIBOR Rate"],
                        mode="lines+markers",
                        name="LIBOR/SOFR Curve",
                        line=dict(color="red"),
                    )
                )
                fig_curves.update_layout(title="Multi-Curve Framework", xaxis_title="Maturity (years)", yaxis_title="Rate (%)")
                st.plotly_chart(fig_curves, use_container_width=True)

                # Basis spread chart
                fig_basis = px.bar(
                    curve_df,
                    x="Maturity",
                    y="Basis Spread",
                    title="Basis Spread (LIBOR - OIS)",
                    labels={"Basis Spread": "Spread (bps)"},
                )
                st.plotly_chart(fig_basis, use_container_width=True)

                # Price comparison
                st.subheader("Single vs Multi-Curve Pricing")
                bond_ids = [b.bond_id for b in bonds[:10]]  # Limit for performance
                selected_id = st.selectbox("Select Bond", bond_ids, key="multi_curve_bond")
                selected_bond = next(b for b in bonds if b.bond_id == selected_id)

                multi_curve_result = multi_curve.price_bond_with_multi_curve(selected_bond)

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Single-Curve Value", format_currency(multi_curve_result["single_curve_value"]))
                with col2:
                    st.metric("Multi-Curve Value", format_currency(multi_curve_result["multi_curve_value"]))
                with col3:
                    st.metric("Difference", format_currency(multi_curve_result["difference"]))
                    st.metric("Difference %", format_percentage(multi_curve_result["difference_pct"]))
                    st.metric("Basis Spread (bps)", f"{multi_curve_result['basis_spread']:.2f}")

    # Tab 9: Portfolio Optimization
    with tab9:
        st.header("Portfolio Optimization")
        st.write("Markowitz, Black-Litterman, and Risk Parity strategies")

        try:
            optimizer = PortfolioOptimizer(valuator)

            opt_method = st.selectbox(
                "Optimization Method", ["Markowitz", "Black-Litterman", "Risk Parity", "Efficient Frontier"], key="opt_method"
            )

            if opt_method == "Markowitz":
                st.subheader("Markowitz Mean-Variance Optimization")

                risk_aversion = st.slider("Risk Aversion", 0.5, 5.0, 1.0, step=0.1, key="risk_aversion")
                target_return = (
                    st.number_input("Target Return (%)", 0.0, 20.0, None, step=0.1) / 100
                    if st.checkbox("Set Target Return", key="target_ret")
                    else None
                )

                try:
                    result = optimizer.markowitz_optimization(bonds, target_return=target_return, risk_aversion=risk_aversion)

                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Portfolio Return", format_percentage(result["portfolio_return"] * 100))
                    with col2:
                        st.metric("Portfolio Volatility", format_percentage(result["portfolio_volatility"] * 100))
                    with col3:
                        st.metric("Sharpe Ratio", f"{result['sharpe_ratio']:.3f}")

                    # Display weights
                    weights_df = pd.DataFrame(
                        {
                            "Bond ID": [b.bond_id for b in bonds],
                            "Weight": [f"{w*100:.2f}%" for w in result["weights"]],
                            "Weight (Decimal)": result["weights"],
                        }
                    )
                    st.dataframe(weights_df, use_container_width=True)

                    # Weight visualization
                    fig_weights = px.bar(weights_df, x="Bond ID", y="Weight (Decimal)", title="Optimal Portfolio Weights")
                    st.plotly_chart(fig_weights, use_container_width=True)
                except Exception as e:
                    st.error(f"Error in Markowitz optimization: {e}")
                    st.info(
                        "This may require additional dependencies (scipy, scikit-learn). Ensure all dependencies are installed."
                    )

            elif opt_method == "Efficient Frontier":
                st.subheader("Efficient Frontier")

                try:
                    with st.spinner("Calculating efficient frontier..."):
                        frontier = optimizer.efficient_frontier(bonds, num_points=30)

                    # Plot efficient frontier
                    frontier_df = pd.DataFrame(
                        {
                            "Return": [r * 100 for r in frontier["returns"]],
                            "Volatility": [v * 100 for v in frontier["volatilities"]],
                            "Sharpe Ratio": frontier["sharpe_ratios"],
                        }
                    )

                    fig_frontier = px.scatter(
                        frontier_df,
                        x="Volatility",
                        y="Return",
                        color="Sharpe Ratio",
                        title="Efficient Frontier",
                        labels={"Volatility": "Volatility (%)", "Return": "Return (%)"},
                    )
                    st.plotly_chart(fig_frontier, use_container_width=True)

                    # Max Sharpe portfolio
                    max_sharpe = frontier["max_sharpe_portfolio"]
                    st.subheader("Maximum Sharpe Ratio Portfolio")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Return", format_percentage(max_sharpe["return"] * 100))
                    with col2:
                        st.metric("Volatility", format_percentage(max_sharpe["volatility"] * 100))
                    with col3:
                        st.metric("Sharpe Ratio", f"{max_sharpe['sharpe_ratio']:.3f}")
                except Exception as e:
                    st.error(f"Error calculating efficient frontier: {e}")
                    st.info("This may require additional dependencies (scipy, scikit-learn).")

            elif opt_method == "Risk Parity":
                st.subheader("Risk Parity Optimization")

                try:
                    result = optimizer.risk_parity_optimization(bonds)

                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Portfolio Volatility", format_percentage(result["portfolio_volatility"] * 100))
                        st.metric("Risk Contribution Std", f"{result['risk_contribution_std']:.4f}")
                    with col2:
                        st.write("**Risk Contributions:**")
                        risk_contrib_df = pd.DataFrame(
                            {
                                "Bond ID": [b.bond_id for b in bonds],
                                "Risk Contribution": [f"{r*100:.2f}%" for r in result["risk_contributions"]],
                            }
                        )
                        st.dataframe(risk_contrib_df, use_container_width=True)
                except Exception as e:
                    st.error(f"Error in Risk Parity optimization: {e}")
                    st.info("This may require additional dependencies (scipy, scikit-learn).")
            elif opt_method == "Black-Litterman":
                st.info("Black-Litterman optimization is not yet implemented in the dashboard.")
        except Exception as e:
            st.error(f"Error in Portfolio Optimization: {e}")
            st.info(
                "This may require additional dependencies (scipy, scikit-learn). Install them with: pip install scipy scikit-learn"
            )

    # Tab 10: Factor Models
    with tab10:
        st.header("Factor Model Analysis")
        st.write("PCA-based factors and risk attribution")

        try:
            factor_model = FactorModel(valuator)

            num_factors = st.slider("Number of Factors", 2, 5, 3, key="num_factors")

            try:
                with st.spinner("Extracting factors..."):
                    factor_result = factor_model.extract_bond_factors(bonds, num_factors=num_factors)

                st.subheader("Factor Analysis")
                col1, col2 = st.columns(2)

                with col1:
                    st.write("**Explained Variance:**")
                    variance_df = pd.DataFrame(
                        {
                            "Factor": [f"Factor {i+1}" for i in range(num_factors)],
                            "Variance %": [f"{v*100:.2f}%" for v in factor_result["explained_variance"]],
                            "Cumulative %": [f"{v*100:.2f}%" for v in factor_result["cumulative_variance"]],
                        }
                    )
                    st.dataframe(variance_df, use_container_width=True)

                with col2:
                    st.write("**Factor Names:**")
                    for i, name in enumerate(factor_result["factor_names"]):
                        st.write(f"Factor {i+1}: {name}")

                # Factor exposures
                st.subheader("Portfolio Factor Exposures")
                exposure_result = factor_model.calculate_factor_exposures(bonds)

                exposure_df = pd.DataFrame(
                    {
                        "Factor": factor_result["factor_names"],
                        "Exposure": [f"{e:.4f}" for e in exposure_result["portfolio_exposures"]],
                        "Contribution": [f"{c*100:.2f}%" for c in exposure_result["factor_contributions"]],
                    }
                )
                st.dataframe(exposure_df, use_container_width=True)

                # Risk attribution
                st.subheader("Risk Attribution")
                risk_attr = factor_model.risk_attribution(bonds)

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Portfolio Volatility", format_percentage(risk_attr["portfolio_volatility"] * 100))
                with col2:
                    st.metric("Factor Risk %", format_percentage(100 - risk_attr["idiosyncratic_risk_pct"]))
                with col3:
                    st.metric("Idiosyncratic Risk %", format_percentage(risk_attr["idiosyncratic_risk_pct"]))

                # Factor risk breakdown
                factor_risk_df = pd.DataFrame(
                    {
                        "Factor": factor_result["factor_names"],
                        "Risk Contribution %": [f"{r:.2f}%" for r in risk_attr["factor_risk_percentages"]],
                    }
                )
                fig_factor_risk = px.bar(
                    factor_risk_df, x="Factor", y="Risk Contribution %", title="Risk Attribution by Factor"
                )
                st.plotly_chart(fig_factor_risk, use_container_width=True)
            except Exception as e:
                st.error(f"Error in factor analysis: {e}")
                st.info("Factor model requires scikit-learn for PCA. Ensure it's installed: pip install scikit-learn")
        except Exception as e:
            st.error(f"Error initializing Factor Model: {e}")
            st.info("Please ensure scikit-learn is installed: pip install scikit-learn")

    # Tab 11: Backtesting & Execution
    with tab11:
        st.header("Backtesting & Execution Strategies")

        exec_subtab = st.radio(
            "Select Feature",
            ["Backtesting", "Execution Strategies", "Correlation Analysis", "Floating Rate Bonds"],
            horizontal=True,
        )

        try:
            if exec_subtab == "Backtesting":
                st.subheader("Strategy Backtesting")
                st.write("Backtest arbitrage detection strategy on historical data")

                # Simplified backtesting with current bonds
                st.info("Note: Full backtesting requires historical bond data. This is a demonstration.")

                try:
                    backtest_engine = BacktestEngine(valuator)

                    initial_capital = st.number_input("Initial Capital", 100000, 10000000, 1000000, step=100000)

                    if st.button("Run Backtest"):
                        # Simulate historical data (in production, would load from database)
                        historical_bonds = [bonds] * 10  # 10 periods

                        try:
                            with st.spinner("Running backtest..."):
                                backtest_result = backtest_engine.backtest_arbitrage_strategy(
                                    historical_bonds, initial_capital=initial_capital
                                )

                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.metric("Total Return", format_percentage(backtest_result["total_return_pct"]))
                            with col2:
                                st.metric("Sharpe Ratio", f"{backtest_result['sharpe_ratio']:.3f}")
                            with col3:
                                st.metric("Max Drawdown", format_percentage(backtest_result["max_drawdown_pct"]))
                            with col4:
                                st.metric("Win Rate", format_percentage(backtest_result["win_rate"] * 100))

                            # Performance chart
                            perf_df = pd.DataFrame(
                                {
                                    "Period": range(len(backtest_result["portfolio_values"])),
                                    "Portfolio Value": backtest_result["portfolio_values"],
                                }
                            )
                            fig_perf = px.line(perf_df, x="Period", y="Portfolio Value", title="Backtest Performance")
                            st.plotly_chart(fig_perf, use_container_width=True)
                        except Exception as e:
                            st.error(f"Error running backtest: {e}")
                except Exception as e:
                    st.error(f"Error initializing BacktestEngine: {e}")

            elif exec_subtab == "Execution Strategies":
                st.subheader("Execution Strategies")

                try:
                    exec_strategy = ExecutionStrategy()

                    bond_ids = [b.bond_id for b in bonds[:10]]
                    selected_id = st.selectbox("Select Bond", bond_ids, key="exec_bond")
                    selected_bond = next(b for b in bonds if b.bond_id == selected_id)

                    strategy_type = st.selectbox("Strategy", ["TWAP", "VWAP", "Optimal Execution"], key="exec_type")
                    total_quantity = st.number_input("Total Quantity", 100, 10000, 1000, step=100)

                    if strategy_type == "TWAP":
                        start_time = datetime.now()
                        end_time = start_time + timedelta(hours=4)
                        num_intervals = st.slider("Number of Intervals", 5, 20, 10)

                        twap_result = exec_strategy.twap_execution(total_quantity, start_time, end_time, num_intervals)

                        st.write("**TWAP Execution Schedule:**")
                        schedule_df = pd.DataFrame(twap_result["schedule"])
                        st.dataframe(
                            schedule_df[["interval", "time", "quantity", "cumulative_quantity"]], use_container_width=True
                        )

                    elif strategy_type == "Optimal Execution":
                        urgency = st.slider("Urgency", 0.0, 1.0, 0.5, step=0.1)
                        volatility = st.slider("Volatility", 0.001, 0.05, 0.01, step=0.001)

                        opt_result = exec_strategy.optimal_execution(selected_bond, total_quantity, urgency, volatility)

                        st.write("**Optimal Execution Schedule:**")
                        schedule_df = pd.DataFrame(opt_result["schedule"])
                        st.dataframe(schedule_df, use_container_width=True)
                except Exception as e:
                    st.error(f"Error in execution strategy: {e}")

            elif exec_subtab == "Correlation Analysis":
                st.subheader("Correlation Analysis")

                try:
                    corr_analyzer = CorrelationAnalyzer(valuator)

                    corr_result = corr_analyzer.calculate_correlation_matrix(bonds)

                    st.metric("Average Correlation", f"{corr_result['average_correlation']:.3f}")
                    st.metric("Diversification Ratio", f"{corr_result['diversification_ratio']:.3f}")

                    # Correlation heatmap
                    corr_df = corr_result["correlation_dataframe"]
                    fig_corr = px.imshow(
                        corr_df.values,
                        labels=dict(x="Bond", y="Bond", color="Correlation"),
                        x=corr_df.columns,
                        y=corr_df.index,
                        title="Bond Correlation Matrix",
                        color_continuous_scale="RdBu",
                    )
                    st.plotly_chart(fig_corr, use_container_width=True)

                    # Diversification metrics
                    st.subheader("Portfolio Diversification Metrics")
                    div_metrics = corr_analyzer.portfolio_diversification_metrics(bonds)

                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Effective Positions", f"{div_metrics['effective_positions']:.2f}")
                    with col2:
                        st.metric("Herfindahl Index", f"{div_metrics['herfindahl_index']:.4f}")
                    with col3:
                        st.metric("Diversification Benefit", format_percentage(div_metrics["diversification_benefit_pct"]))
                except Exception as e:
                    st.error(f"Error in correlation analysis: {e}")

            elif exec_subtab == "Floating Rate Bonds":
                st.subheader("Floating Rate Bond Analysis")

                floating_bonds = [b for b in bonds if b.bond_type == BondType.FLOATING_RATE]

                if floating_bonds:
                    fr_pricer = FloatingRateBondPricer(valuator)

                    bond_ids = [b.bond_id for b in floating_bonds]
                    selected_id = st.selectbox("Select Floating Rate Bond", bond_ids, key="fr_bond")
                    selected_bond = next(b for b in floating_bonds if b.bond_id == selected_id)

                    next_reset = st.date_input("Next Reset Date", value=datetime.now().date() + timedelta(days=182))
                    spread = st.slider("Spread (bps)", -50, 200, 0, step=5) / 10000

                    fr_result = fr_pricer.price_floating_rate_bond(
                        selected_bond, datetime.combine(next_reset, datetime.min.time()), spread=spread
                    )

                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Clean Price", format_currency(fr_result["clean_price"]))
                        st.metric("Dirty Price", format_currency(fr_result["dirty_price"]))
                    with col2:
                        st.metric("Next Coupon Rate", format_percentage(fr_result["next_coupon_rate"]))
                        st.metric("Reference Rate", format_percentage(fr_result["reference_rate"]))
                    with col3:
                        st.metric("Price to Par", f"{fr_result['price_to_par']:.4f}")
                        st.metric("Accrued Interest", format_currency(fr_result["accrued_interest"]))

                    # Discount Margin
                    dm_result = fr_pricer.calculate_discount_margin(
                        selected_bond, selected_bond.current_price, datetime.combine(next_reset, datetime.min.time())
                    )
                    if "error" not in dm_result:
                        st.metric("Discount Margin", f"{dm_result['discount_margin_bps']:.2f} bps")
                else:
                    st.info("No floating rate bonds found. Generate bonds with FLOATING_RATE type.")
        except Exception as e:
            st.error(f"Error in backtesting/execution tab: {e}")

    # Tab 12: Advanced ML & AI
    with tab12:
        st.header("Advanced Machine Learning & AI")
        st.write("Beyond industry standards: Deep learning, explainable AI, AutoML")

        adv_ml_subtab = st.radio(
            "Select Feature",
            ["Ensemble ML", "AutoML", "Explainable AI", "Tail Risk", "Regime Detection", "Alternative Data"],
            horizontal=True,
        )

        try:
            if adv_ml_subtab == "Ensemble ML":
                st.subheader("Advanced Ensemble Machine Learning")

                try:
                    advanced_ml = AdvancedMLBondAdjuster(valuator)

                    if st.button("Train Ensemble Model", key="train_ensemble"):
                        with st.spinner("Training ensemble (this may take a minute)..."):
                            try:
                                ensemble_result = advanced_ml.train_ensemble(
                                    bonds, test_size=config.ml_test_size, random_state=config.ml_random_state
                                )

                                st.success("Ensemble model trained!")

                                st.subheader("Model Performance Comparison")
                                model_df = pd.DataFrame(
                                    [
                                        {
                                            "Model": name,
                                            "Test R²": f"{metrics['test_r2']:.4f}",
                                            "Test RMSE": f"{metrics['test_rmse']:.4f}",
                                        }
                                        for name, metrics in ensemble_result["individual_models"].items()
                                    ]
                                )
                                model_df.loc[len(model_df)] = {
                                    "Model": "Ensemble (Stacking)",
                                    "Test R²": f"{ensemble_result['ensemble_metrics']['test_r2']:.4f}",
                                    "Test RMSE": f"{ensemble_result['ensemble_metrics']['test_rmse']:.4f}",
                                }
                                st.dataframe(model_df, use_container_width=True)

                                st.metric("Ensemble Improvement", f"{ensemble_result['improvement_over_best']:.4f} R²")

                                # Feature importance
                                st.subheader("Feature Importance (Explainable AI)")
                                importance = advanced_ml.get_feature_importance_explained()
                                if "sorted" in importance:
                                    top_features = list(importance["sorted"].items())[:10]
                                    feature_df = pd.DataFrame(top_features, columns=["Feature", "Importance"])
                                    fig_importance = px.bar(
                                        feature_df, x="Feature", y="Importance", title="Top 10 Most Important Features"
                                    )
                                    st.plotly_chart(fig_importance, use_container_width=True)
                            except Exception as e:
                                st.error(f"Training error: {e}")
                                st.info(
                                    "ML features may require scikit-learn, xgboost, lightgbm, or catboost. Some libraries may not be available due to dependency issues."
                                )

                    # Prediction with uncertainty
                    st.subheader("Prediction with Uncertainty Quantification")
                    bond_ids = [b.bond_id for b in bonds[:10]]
                    selected_id = st.selectbox("Select Bond", bond_ids, key="uncertainty_bond")
                    selected_bond = next(b for b in bonds if b.bond_id == selected_id)

                    if advanced_ml.is_trained:
                        try:
                            uncertainty_result = advanced_ml.predict_with_uncertainty(selected_bond)

                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("ML-Adjusted Value", format_currency(uncertainty_result["mean_ml_value"]))
                            with col2:
                                st.metric(
                                    "Lower Bound (95% CI)", format_currency(uncertainty_result["confidence_interval_lower"])
                                )
                            with col3:
                                st.metric(
                                    "Upper Bound (95% CI)", format_currency(uncertainty_result["confidence_interval_upper"])
                                )
                                st.metric("Uncertainty %", format_percentage(uncertainty_result["uncertainty_pct"]))
                        except Exception as e:
                            st.error(f"Error in prediction: {e}")
                    else:
                        st.info("Train the ensemble model first to see predictions with uncertainty quantification.")
                except Exception as e:
                    st.error(f"Error initializing Advanced ML: {e}")
                    st.info("Advanced ML features require scikit-learn. Ensure it's installed: pip install scikit-learn")

            elif adv_ml_subtab == "AutoML":
                st.subheader("Automated Machine Learning (AutoML)")
                st.write("Automatically selects best model and hyperparameters")

                try:
                    automl = AutoMLBondAdjuster(valuator)

                    if st.button("Run AutoML", key="run_automl"):
                        with st.spinner("Running AutoML (evaluating multiple models)..."):
                            try:
                                automl_result = automl.automated_model_selection(bonds, max_evaluation_time=300)

                                st.success(f"Best model selected: {automl_result['best_model']}")

                                col1, col2 = st.columns(2)
                                with col1:
                                    st.metric("Best Model", automl_result["best_model"])
                                    st.metric("Best Score (R²)", f"{automl_result['best_score']:.4f}")
                                with col2:
                                    st.write("**Best Hyperparameters:**")
                                    for param, value in automl_result["best_params"].items():
                                        st.write(f"- {param}: {value}")

                                # Model comparison
                                st.subheader("Model Comparison")
                                comparison_df = pd.DataFrame(
                                    [
                                        {"Model": name, "R² Score": f"{score:.4f}"}
                                        for name, score in automl_result["all_models"].items()
                                    ]
                                )
                                fig_comparison = px.bar(
                                    comparison_df, x="Model", y="R² Score", title="AutoML Model Comparison"
                                )
                                st.plotly_chart(fig_comparison, use_container_width=True)
                            except Exception as e:
                                st.error(f"AutoML error: {e}")
                                st.info("AutoML requires scikit-learn and may require additional ML libraries.")
                except Exception as e:
                    st.error(f"Error initializing AutoML: {e}")

            elif adv_ml_subtab == "Explainable AI":
                st.subheader("Explainable AI - Model Interpretability")

                try:
                    advanced_ml = AdvancedMLBondAdjuster(valuator)

                    if not advanced_ml.is_trained:
                        if st.button("Train Model for Explanation", key="train_explain"):
                            with st.spinner("Training model..."):
                                try:
                                    advanced_ml.train_ensemble(
                                        bonds, test_size=config.ml_test_size, random_state=config.ml_random_state
                                    )
                                except Exception as e:
                                    st.error(f"Training error: {e}")

                    if advanced_ml.is_trained:
                        bond_ids = [b.bond_id for b in bonds]
                        selected_id = st.selectbox("Select Bond", bond_ids, key="explain_bond")
                        selected_bond = next(b for b in bonds if b.bond_id == selected_id)

                        try:
                            explain_result = advanced_ml.explain_prediction(selected_bond)

                            st.subheader(f"Prediction Explanation for {selected_bond.bond_id}")
                            col1, col2 = st.columns(2)
                            with col1:
                                st.metric("Theoretical Value", format_currency(explain_result["theoretical_fair_value"]))
                                st.metric("ML-Adjusted Value", format_currency(explain_result["ml_adjusted_value"]))
                                st.metric("Adjustment Factor", f"{explain_result['adjustment_factor']:.4f}")

                            with col2:
                                st.write("**Top 5 Feature Drivers:**")
                                for feature, contribution in explain_result["top_drivers"]:
                                    st.write(f"- **{feature}**: {contribution:.4f}")
                        except Exception as e:
                            st.error(f"Error explaining prediction: {e}")
                    else:
                        st.info("Train the model first to see explainable AI features.")
                except Exception as e:
                    st.error(f"Error in Explainable AI: {e}")

            elif adv_ml_subtab == "Tail Risk":
                st.subheader("Advanced Tail Risk Analysis (CVaR, Expected Shortfall)")

                try:
                    tail_risk = TailRiskAnalyzer(valuator)

                    confidence_level = st.slider("Confidence Level", 0.90, 0.999, 0.95, step=0.01)

                    try:
                        cvar_result = tail_risk.calculate_cvar(bonds, confidence_level=confidence_level)

                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("VaR", format_currency(cvar_result["var_value"]))
                            st.metric("VaR %", format_percentage(cvar_result["var_pct"]))
                        with col2:
                            st.metric("CVaR (Expected Shortfall)", format_currency(cvar_result["cvar_value"]))
                            st.metric("CVaR %", format_percentage(cvar_result["cvar_pct"]))
                        with col3:
                            st.metric("Tail Ratio (CVaR/VaR)", f"{cvar_result['tail_ratio']:.3f}")
                            st.write(f"*Higher ratio = heavier tail*")

                        # Multiple confidence levels
                        st.subheader("Expected Shortfall at Multiple Levels")
                        es_levels = tail_risk.calculate_expected_shortfall_multiple_levels(
                            bonds, confidence_levels=[0.90, 0.95, 0.99, 0.999]
                        )

                        es_df = pd.DataFrame(
                            [
                                {
                                    "Confidence Level": level,
                                    "Expected Shortfall %": f"{data['es_pct']:.2f}%",
                                    "VaR %": f"{data['var_pct']:.2f}%",
                                    "Tail Ratio": f"{data['tail_ratio']:.3f}",
                                }
                                for level, data in es_levels.items()
                            ]
                        )
                        st.dataframe(es_df, use_container_width=True)
                    except Exception as e:
                        st.error(f"Error calculating tail risk: {e}")
                except Exception as e:
                    st.error(f"Error initializing Tail Risk Analyzer: {e}")

            elif adv_ml_subtab == "Regime Detection":
                st.subheader("Market Regime Detection & Adaptive Models")

                try:
                    regime_detector = RegimeDetector(valuator)

                    num_regimes = st.slider("Number of Regimes", 2, 5, 3)

                    try:
                        with st.spinner("Detecting market regimes..."):
                            regime_result = regime_detector.detect_regimes(bonds, num_regimes=num_regimes)

                        st.subheader("Detected Regimes")
                        regime_df = pd.DataFrame(
                            [
                                {
                                    "Regime": name,
                                    "Num Bonds": data["num_bonds"],
                                    "Avg YTM": format_percentage(data["avg_ytm"]),
                                    "Avg Spread (bps)": f"{data['avg_spread_bps']:.1f}",
                                    "Avg Duration": f"{data['avg_duration']:.2f}",
                                    "Regime Type": data["regime_type"],
                                }
                                for name, data in regime_result["regime_analysis"].items()
                            ]
                        )
                        st.dataframe(regime_df, use_container_width=True)

                        # Regime-dependent pricing
                        st.subheader("Regime-Dependent Pricing")
                        bond_ids = [b.bond_id for b in bonds]
                        selected_id = st.selectbox("Select Bond", bond_ids, key="regime_bond")
                        selected_bond = next(b for b in bonds if b.bond_id == selected_id)

                        regime_pricing = regime_detector.regime_dependent_pricing(selected_bond)

                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Current Regime", f"Regime {regime_pricing['current_regime']}")
                            st.metric("Base Fair Value", format_currency(regime_pricing["base_fair_value"]))
                        with col2:
                            st.metric("Regime Adjustment", format_percentage(regime_pricing["adjustment_pct"]))
                            st.metric("Regime-Adjusted Value", format_currency(regime_pricing["regime_adjusted_value"]))
                    except Exception as e:
                        st.error(f"Error in regime detection: {e}")
                except Exception as e:
                    st.error(f"Error initializing Regime Detector: {e}")

            elif adv_ml_subtab == "Alternative Data":
                st.subheader("Alternative Data Integration")
                st.write("ESG factors, sentiment analysis, economic indicators")

                try:
                    alt_data = AlternativeDataAnalyzer(valuator)

                    data_type = st.selectbox(
                        "Data Type", ["ESG Analysis", "Sentiment Analysis", "Economic Factors"], key="alt_data_type"
                    )

                    bond_ids = [b.bond_id for b in bonds]
                    selected_id = st.selectbox("Select Bond", bond_ids, key="alt_data_bond")
                    selected_bond = next(b for b in bonds if b.bond_id == selected_id)

                    try:
                        if data_type == "ESG Analysis":
                            esg_result = alt_data.calculate_esg_score(selected_bond)

                            st.subheader(f"ESG Analysis for {selected_bond.bond_id}")
                            col1, col2 = st.columns(2)
                            with col1:
                                st.metric("Overall ESG Score", f"{esg_result['esg_score']:.1f}/100")
                                st.metric("ESG Rating", esg_result["esg_rating"])
                            with col2:
                                st.metric("Environmental Score", f"{esg_result['environmental_score']}/100")
                                st.metric("Social Score", f"{esg_result['social_score']}/100")
                                st.metric("Governance Score", f"{esg_result['governance_score']}/100")

                            col1, col2 = st.columns(2)
                            with col1:
                                st.metric("Base Fair Value", format_currency(esg_result["base_fair_value"]))
                            with col2:
                                st.metric("ESG-Adjusted Value", format_currency(esg_result["esg_adjusted_value"]))
                                st.metric("ESG Impact", format_percentage(esg_result["esg_impact_pct"]))

                            # ESG breakdown
                            fig_esg = go.Figure()
                            fig_esg.add_trace(
                                go.Bar(
                                    x=["Environmental", "Social", "Governance"],
                                    y=[
                                        esg_result["environmental_score"],
                                        esg_result["social_score"],
                                        esg_result["governance_score"],
                                    ],
                                    marker_color=["green", "blue", "purple"],
                                )
                            )
                            fig_esg.update_layout(title="ESG Score Breakdown", yaxis_title="Score (0-100)")
                            st.plotly_chart(fig_esg, use_container_width=True)

                        elif data_type == "Sentiment Analysis":
                            sentiment_score = st.slider("News Sentiment (-1 to 1)", -1.0, 1.0, 0.0, step=0.1)

                            sentiment_result = alt_data.sentiment_analysis(selected_bond, sentiment_score)

                            st.subheader("Sentiment Impact Analysis")
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Sentiment Score", f"{sentiment_result['news_sentiment']:.2f}")
                                st.metric("Sentiment Label", sentiment_result["sentiment_label"])
                            with col2:
                                st.metric("Base Fair Value", format_currency(sentiment_result["base_fair_value"]))
                            with col3:
                                st.metric(
                                    "Sentiment-Adjusted Value", format_currency(sentiment_result["sentiment_adjusted_value"])
                                )
                                st.metric("Impact %", format_percentage(sentiment_result["sentiment_impact_pct"]))

                        elif data_type == "Economic Factors":
                            st.subheader("Economic Factor Impact")

                            inflation = st.slider("Inflation Expectation (%)", 0.0, 10.0, 2.0, step=0.1) / 100
                            gdp = st.slider("GDP Growth (%)", -5.0, 10.0, 3.0, step=0.1) / 100
                            unemployment = st.slider("Unemployment Rate (%)", 2.0, 15.0, 4.0, step=0.1) / 100

                            econ_result = alt_data.economic_factors_impact([selected_bond], inflation, gdp, unemployment)

                            if econ_result["bond_impacts"]:
                                impact = econ_result["bond_impacts"][0]
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.metric("Inflation Impact", format_percentage(impact["inflation_impact_pct"]))
                                with col2:
                                    st.metric("GDP Impact", format_percentage(impact["gdp_impact_pct"]))
                                with col3:
                                    st.metric("Unemployment Impact", format_percentage(impact["unemployment_impact_pct"]))

                                st.metric("Total Economic Impact", format_percentage(impact["total_impact_pct"]))
                                st.metric("Adjusted Value", format_currency(impact["adjusted_value"]))
                    except Exception as e:
                        st.error(f"Error in Alternative Data analysis: {e}")
                except Exception as e:
                    st.error(f"Error initializing Alternative Data Analyzer: {e}")
        except Exception as e:
            st.error(f"Error in Advanced ML & AI tab: {e}")
            st.info(
                "Some ML features may require additional dependencies. Check that scikit-learn is installed: pip install scikit-learn"
            )


if __name__ == "__main__":
    main()
