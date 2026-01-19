"""
Comprehensive Training Dataset Generator
Follows best practices from leading financial firms for bond model training

Key Features:
- Large, diverse dataset (thousands of bonds)
- Multiple market regimes (bull, bear, high/low volatility)
- Time series data across economic cycles
- Realistic market microstructure
- Proper train/validation/test splits
- Data quality validation
- Stress testing scenarios
"""

import os
import random
import warnings
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from bondtrader.core.bond_models import Bond, BondType
from bondtrader.core.bond_valuation import BondValuator
from bondtrader.data.data_generator import BondDataGenerator


@dataclass
class MarketRegime:
    """Market regime characteristics"""

    regime_name: str
    risk_free_rate: float
    volatility_multiplier: float
    credit_spread_base: float
    liquidity_factor: float
    market_sentiment: float  # -1 to 1, negative = bear, positive = bull


class TrainingDataGenerator:
    """
    Comprehensive training data generator following financial industry best practices

    Best Practices Implemented:
    1. Large sample size (thousands of observations)
    2. Multiple market regimes and economic cycles
    3. Time series data (not just cross-sectional)
    4. Realistic market microstructure
    5. Proper data splits (train/validation/test)
    6. Data quality validation
    7. Stress testing scenarios
    8. Feature engineering
    9. Out-of-sample validation periods
    10. Multiple bond types and credit ratings
    """

    def __init__(self, seed: int = 42) -> None:
        """
        Initialize generator with seed for reproducibility

        Args:
            seed: Random seed for reproducibility (default: 42)
        """
        self.seed = seed
        np.random.seed(seed)
        random.seed(seed)
        self.valuator = BondValuator()
        self.base_generator = BondDataGenerator(seed=seed)

        # Define market regimes (following industry practice)
        self.regimes = {
            "normal": MarketRegime(
                regime_name="Normal Market",
                risk_free_rate=0.03,
                volatility_multiplier=1.0,
                credit_spread_base=0.0,
                liquidity_factor=1.0,
                market_sentiment=0.0,
            ),
            "bull": MarketRegime(
                regime_name="Bull Market",
                risk_free_rate=0.025,
                volatility_multiplier=0.7,
                credit_spread_base=-0.005,  # Tighter spreads
                liquidity_factor=1.2,
                market_sentiment=0.7,
            ),
            "bear": MarketRegime(
                regime_name="Bear Market",
                risk_free_rate=0.04,
                volatility_multiplier=1.5,
                credit_spread_base=0.015,  # Wider spreads
                liquidity_factor=0.7,
                market_sentiment=-0.6,
            ),
            "high_volatility": MarketRegime(
                regime_name="High Volatility",
                risk_free_rate=0.035,
                volatility_multiplier=2.0,
                credit_spread_base=0.010,
                liquidity_factor=0.6,
                market_sentiment=-0.3,
            ),
            "low_volatility": MarketRegime(
                regime_name="Low Volatility",
                risk_free_rate=0.025,
                volatility_multiplier=0.5,
                credit_spread_base=-0.003,
                liquidity_factor=1.3,
                market_sentiment=0.4,
            ),
            "crisis": MarketRegime(
                regime_name="Financial Crisis",
                risk_free_rate=0.02,
                volatility_multiplier=3.0,
                credit_spread_base=0.050,  # Very wide spreads
                liquidity_factor=0.4,
                market_sentiment=-0.9,
            ),
            "recovery": MarketRegime(
                regime_name="Recovery",
                risk_free_rate=0.015,
                volatility_multiplier=1.2,
                credit_spread_base=0.005,
                liquidity_factor=0.9,
                market_sentiment=0.2,
            ),
        }

    def generate_comprehensive_dataset(
        self,
        total_bonds: int = 5000,
        time_periods: int = 60,  # 60 months = 5 years
        bonds_per_period: int = 100,
        train_split: float = 0.7,
        validation_split: float = 0.15,
        test_split: float = 0.15,
    ) -> Dict:
        """
        Generate comprehensive training dataset

        Industry Best Practices:
        - Large sample size (5000+ bonds)
        - Multiple time periods (5+ years)
        - Multiple market regimes
        - Proper train/validation/test splits
        - Time-based splits (not random) to prevent look-ahead bias

        Args:
            total_bonds: Total number of unique bonds
            time_periods: Number of time periods (months)
            bonds_per_period: Bonds observed per period
            train_split: Training set proportion
            validation_split: Validation set proportion
            test_split: Test set proportion

        Returns:
            Dictionary with training datasets and metadata
        """
        if abs(train_split + validation_split + test_split - 1.0) > 1e-6:
            raise ValueError("Splits must sum to 1.0")

        print("Generating comprehensive training dataset...")
        print(f"  Total bonds: {total_bonds}")
        print(f"  Time periods: {time_periods}")
        print(f"  Bonds per period: {bonds_per_period}")

        # Generate base bond universe
        print("  Step 1: Generating base bond universe...")
        base_bonds = self._generate_diverse_bond_universe(total_bonds)

        # Generate time series data across regimes
        print("  Step 2: Generating time series data across market regimes...")
        time_series_data = self._generate_time_series_data(base_bonds, time_periods, bonds_per_period)

        # Create feature matrices
        print("  Step 3: Creating feature matrices...")
        features, targets, metadata = self._create_feature_matrices(time_series_data)

        # Split data (time-based, not random - critical for financial data)
        print("  Step 4: Creating time-based splits...")
        splits = self._create_time_based_splits(features, targets, metadata, train_split, validation_split, test_split)

        # Data quality validation
        print("  Step 5: Validating data quality...")
        quality_report = self._validate_data_quality(splits)

        # Generate stress test scenarios
        print("  Step 6: Generating stress test scenarios...")
        stress_scenarios = self._generate_stress_scenarios(base_bonds)

        print("  ✓ Dataset generation complete!")

        return {
            "train": {
                "bonds": splits["train"]["bonds"],
                "features": splits["train"]["features"],
                "targets": splits["train"]["targets"],
                "metadata": splits["train"]["metadata"],
            },
            "validation": {
                "bonds": splits["validation"]["bonds"],
                "features": splits["validation"]["features"],
                "targets": splits["validation"]["targets"],
                "metadata": splits["validation"]["metadata"],
            },
            "test": {
                "bonds": splits["test"]["bonds"],
                "features": splits["test"]["features"],
                "targets": splits["test"]["targets"],
                "metadata": splits["test"]["metadata"],
            },
            "stress_scenarios": stress_scenarios,
            "quality_report": quality_report,
            "dataset_metadata": {
                "total_bonds": total_bonds,
                "time_periods": time_periods,
                "total_observations": len(features),
                "train_size": len(splits["train"]["features"]),
                "validation_size": len(splits["validation"]["features"]),
                "test_size": len(splits["test"]["features"]),
                "num_features": features.shape[1] if len(features) > 0 else 0,
                "regimes_represented": list(set([m["regime"] for m in metadata])),
                "date_range": {"start": min([m["date"] for m in metadata]), "end": max([m["date"] for m in metadata])},
            },
        }

    def _generate_diverse_bond_universe(self, num_bonds: int) -> List[Bond]:
        """
        Generate diverse bond universe

        Industry Best Practice: Ensure representation across:
        - Credit ratings (AAA to CCC)
        - Maturities (short, medium, long)
        - Bond types (Treasury, Corporate, High Yield, etc.)
        - Coupon structures
        """
        bonds = []

        # Credit rating distribution (realistic market distribution)
        ratings = [
            "AAA",
            "AA+",
            "AA",
            "AA-",
            "A+",
            "A",
            "A-",
            "BBB+",
            "BBB",
            "BBB-",
            "BB+",
            "BB",
            "BB-",
            "B+",
            "B",
            "B-",
            "CCC+",
            "CCC",
            "CCC-",
        ]
        rating_probs = [
            0.05,
            0.05,
            0.08,
            0.07,
            0.10,
            0.12,
            0.10,
            0.08,
            0.10,
            0.07,
            0.05,
            0.04,
            0.03,
            0.02,
            0.02,
            0.01,
            0.005,
            0.005,
            0.005,
        ]
        # Normalize probabilities to sum to 1.0
        total_prob = sum(rating_probs)
        rating_probs = [p / total_prob for p in rating_probs]

        # Bond type distribution
        bond_types = [BondType.TREASURY, BondType.CORPORATE, BondType.HIGH_YIELD, BondType.FIXED_RATE, BondType.ZERO_COUPON]
        type_probs = [0.20, 0.40, 0.15, 0.20, 0.05]

        # Maturity distribution (years)
        maturity_ranges = [
            (0.5, 2),  # Short-term: 20%
            (2, 5),  # Medium-term: 30%
            (5, 10),  # Medium-long: 30%
            (10, 20),  # Long-term: 15%
            (20, 30),  # Very long: 5%
        ]
        maturity_probs = [0.20, 0.30, 0.30, 0.15, 0.05]

        issuers = [
            "US Treasury",
            "Apple Inc",
            "Microsoft Corp",
            "JPMorgan Chase",
            "Bank of America",
            "Goldman Sachs",
            "Exxon Mobil",
            "AT&T Inc",
            "Verizon Communications",
            "Coca-Cola Co",
            "Walmart Inc",
            "Amazon.com Inc",
            "Google LLC",
            "Meta Platforms",
            "Tesla Inc",
            "General Electric",
            "Ford Motor Co",
            "General Motors",
            "Boeing Co",
            "Lockheed Martin",
            "Raytheon Technologies",
        ]

        for i in range(num_bonds):
            # Select characteristics
            rating = np.random.choice(ratings, p=rating_probs)
            bond_type = np.random.choice(bond_types, p=type_probs)

            # Maturity
            maturity_range = np.random.choice(len(maturity_ranges), p=maturity_probs)
            min_maturity, max_maturity = maturity_ranges[maturity_range]
            time_to_maturity = np.random.uniform(min_maturity, max_maturity)

            # Issue date (bonds issued over past 10 years)
            years_since_issue = np.random.uniform(0, 10)
            issue_date = datetime.now() - timedelta(days=int(years_since_issue * 365.25))
            maturity_date = datetime.now() + timedelta(days=int(time_to_maturity * 365.25))

            # Face value
            face_value = np.random.choice([1000, 5000, 10000, 25000, 100000])

            # Coupon rate (based on type and rating)
            if bond_type == BondType.ZERO_COUPON:
                coupon_rate = 0.0
            elif bond_type == BondType.TREASURY:
                coupon_rate = np.random.uniform(1.5, 4.5)
            elif bond_type == BondType.HIGH_YIELD:
                coupon_rate = np.random.uniform(6.0, 12.0)
            else:
                # Corporate bonds: higher rating = lower coupon
                base_coupon = 2.0 if rating.startswith("A") else 3.0 if rating.startswith("BBB") else 5.0
                coupon_rate = np.random.uniform(base_coupon, base_coupon + 2.0)

            # Issuer
            if bond_type == BondType.TREASURY:
                issuer = "US Treasury"
            else:
                issuer = np.random.choice(issuers)

            # Additional features
            frequency = 2  # Semi-annual
            callable = np.random.random() < 0.2
            convertible = np.random.random() < 0.1

            # Initial price (will be adjusted by regime)
            base_price_ratio = np.random.uniform(0.90, 1.10)
            current_price = face_value * base_price_ratio

            bond_id = f"BOND-{i+1:06d}-{bond_type.value[:3].upper()}-{rating}"

            try:
                bond = Bond(
                    bond_id=bond_id,
                    bond_type=bond_type,
                    face_value=face_value,
                    coupon_rate=coupon_rate,
                    maturity_date=maturity_date,
                    issue_date=issue_date,
                    current_price=current_price,
                    credit_rating=rating,
                    issuer=issuer,
                    frequency=frequency,
                    callable=callable,
                    convertible=convertible,
                )
                bonds.append(bond)
            except ValueError:
                continue

        return bonds

    def _generate_time_series_data(self, base_bonds: List[Bond], time_periods: int, bonds_per_period: int) -> List[Dict]:
        """
        Generate time series data across multiple market regimes

        Industry Best Practice: Time series data prevents look-ahead bias
        and allows models to learn regime-dependent patterns
        """
        time_series_data = []

        # Regime transition probabilities (Markov chain)
        regime_transitions = {
            "normal": {"normal": 0.6, "bull": 0.15, "bear": 0.15, "high_volatility": 0.05, "low_volatility": 0.05},
            "bull": {"normal": 0.3, "bull": 0.5, "bear": 0.1, "high_volatility": 0.05, "low_volatility": 0.05},
            "bear": {"normal": 0.2, "bull": 0.1, "bear": 0.5, "high_volatility": 0.15, "crisis": 0.05},
            "high_volatility": {"normal": 0.3, "bear": 0.3, "high_volatility": 0.3, "crisis": 0.1},
            "low_volatility": {"normal": 0.4, "bull": 0.4, "low_volatility": 0.2},
            "crisis": {"recovery": 0.4, "bear": 0.3, "high_volatility": 0.2, "crisis": 0.1},
            "recovery": {"normal": 0.5, "bull": 0.3, "recovery": 0.2},
        }

        current_regime = "normal"
        start_date = datetime.now() - timedelta(days=time_periods * 30)

        for period in range(time_periods):
            # Transition to new regime
            if period > 0:
                transition_probs = regime_transitions.get(current_regime, regime_transitions["normal"])
                current_regime = np.random.choice(list(transition_probs.keys()), p=list(transition_probs.values()))

            regime = self.regimes[current_regime]
            period_date = start_date + timedelta(days=period * 30)

            # Sample bonds for this period
            sampled_bonds = random.sample(base_bonds, min(bonds_per_period, len(base_bonds)))

            # Update valuator for this regime
            self.valuator.risk_free_rate = regime.risk_free_rate

            for bond in sampled_bonds:
                # Calculate fair value under current regime
                # Adjust credit spread based on regime
                base_spread = self.valuator._get_credit_spread(bond.credit_rating)
                adjusted_spread = base_spread + regime.credit_spread_base

                # Calculate fair value with regime-adjusted spread
                required_ytm = regime.risk_free_rate + adjusted_spread
                fair_value = self.valuator.calculate_fair_value(bond, required_yield=required_ytm)

                # Add market microstructure noise (bid-ask, liquidity)
                volatility = 0.02 * regime.volatility_multiplier
                liquidity_noise = np.random.normal(0, volatility / regime.liquidity_factor)
                sentiment_impact = regime.market_sentiment * 0.01

                # Market price with regime effects
                market_price = fair_value * (1 + liquidity_noise + sentiment_impact + np.random.normal(0, volatility * 0.5))

                # Ensure price is reasonable
                market_price = np.clip(market_price, fair_value * 0.5, fair_value * 1.5)
                bond.current_price = market_price

                time_series_data.append(
                    {
                        "bond": bond,
                        "date": period_date,
                        "regime": current_regime,
                        "fair_value": fair_value,
                        "market_price": market_price,
                        "risk_free_rate": regime.risk_free_rate,
                        "volatility": volatility,
                        "liquidity_factor": regime.liquidity_factor,
                        "sentiment": regime.market_sentiment,
                    }
                )

        return time_series_data

    def _create_feature_matrices(self, time_series_data: List[Dict]) -> Tuple[np.ndarray, np.ndarray, List[Dict]]:
        """
        Create feature matrices for ML models

        Features include:
        - Bond characteristics
        - Market regime indicators
        - Time features
        - Derived metrics (duration, convexity, etc.)
        """
        features = []
        targets = []
        metadata = []

        for data_point in time_series_data:
            bond = data_point["bond"]
            fair_value = data_point["fair_value"]
            market_price = data_point["market_price"]

            # Calculate bond metrics
            ytm = self.valuator.calculate_yield_to_maturity(bond)
            duration = self.valuator.calculate_duration(bond, ytm)
            convexity = self.valuator.calculate_convexity(bond, ytm)

            char = bond.get_bond_characteristics()

            # Feature vector (aligned with model expectations)
            feature_vector = [
                char["coupon_rate"],
                char["time_to_maturity"],
                char["credit_rating_numeric"],
                char["current_price"] / char["face_value"],  # Price to par
                char["years_since_issue"],
                char["frequency"],
                char["callable"],
                char["convertible"],
                ytm * 100,
                duration,
                convexity,
                market_price / fair_value if fair_value > 0 else 1.0,  # Price to fair ratio
                bond.face_value,
                duration / (1 + ytm) if ytm > 0 else duration,  # Modified duration
                ytm - data_point["risk_free_rate"],  # Spread over RF
                # Regime features
                data_point["risk_free_rate"] * 100,
                data_point["volatility"] * 100,
                data_point["liquidity_factor"],
                data_point["sentiment"],
                # Time features
                data_point["date"].month / 12.0,  # Normalized month
                data_point["date"].year - 2020,  # Years since 2020
                # Regime indicators (one-hot encoded)
                1.0 if data_point["regime"] == "normal" else 0.0,
                1.0 if data_point["regime"] == "bull" else 0.0,
                1.0 if data_point["regime"] == "bear" else 0.0,
                1.0 if data_point["regime"] == "high_volatility" else 0.0,
                1.0 if data_point["regime"] == "low_volatility" else 0.0,
                1.0 if data_point["regime"] == "crisis" else 0.0,
                1.0 if data_point["regime"] == "recovery" else 0.0,
                # Polynomial features
                char["coupon_rate"] ** 2,
                char["time_to_maturity"] ** 2,
                duration**2,
                char["coupon_rate"] * char["time_to_maturity"],
                char["coupon_rate"] * duration,
                char["time_to_maturity"] * duration,
            ]

            features.append(feature_vector)

            # Target: adjustment factor (market_price / fair_value)
            target = market_price / fair_value if fair_value > 0 else 1.0
            targets.append(target)

            metadata.append(
                {
                    "bond_id": bond.bond_id,
                    "date": data_point["date"],
                    "regime": data_point["regime"],
                    "credit_rating": bond.credit_rating,
                    "bond_type": bond.bond_type.value,
                    "issuer": bond.issuer,
                    "coupon_rate": bond.coupon_rate,
                    "face_value": bond.face_value,
                    "maturity_date": bond.maturity_date,
                    "issue_date": bond.issue_date,
                    "frequency": bond.frequency,
                    "callable": bond.callable,
                    "convertible": bond.convertible,
                }
            )

        return np.array(features), np.array(targets), metadata

    def _create_time_based_splits(
        self,
        features: np.ndarray,
        targets: np.ndarray,
        metadata: List[Dict],
        train_split: float,
        validation_split: float,
        test_split: float,
    ) -> Dict:
        """
        Create time-based splits (not random)

        Industry Best Practice: Time-based splits prevent look-ahead bias
        - Train: earliest periods
        - Validation: middle periods
        - Test: latest periods (most recent)
        """
        n = len(features)

        # Sort by date
        sorted_indices = sorted(range(n), key=lambda i: metadata[i]["date"])

        train_end = int(n * train_split)
        validation_end = train_end + int(n * validation_split)

        train_indices = sorted_indices[:train_end]
        validation_indices = sorted_indices[train_end:validation_end]
        test_indices = sorted_indices[validation_end:]

        return {
            "train": {
                "bonds": [metadata[i]["bond_id"] for i in train_indices],
                "features": features[train_indices],
                "targets": targets[train_indices],
                "metadata": [metadata[i] for i in train_indices],
            },
            "validation": {
                "bonds": [metadata[i]["bond_id"] for i in validation_indices],
                "features": features[validation_indices],
                "targets": targets[validation_indices],
                "metadata": [metadata[i] for i in validation_indices],
            },
            "test": {
                "bonds": [metadata[i]["bond_id"] for i in test_indices],
                "features": features[test_indices],
                "targets": targets[test_indices],
                "metadata": [metadata[i] for i in test_indices],
            },
        }

    def _validate_data_quality(self, splits: Dict) -> Dict:
        """
        Validate data quality

        Industry Best Practice: Comprehensive data quality checks
        """
        quality_checks = {}

        for split_name, split_data in splits.items():
            features = split_data["features"]
            targets = split_data["targets"]

            checks = {
                "n_samples": len(features),
                "n_features": features.shape[1] if len(features) > 0 else 0,
                "missing_values": np.isnan(features).sum(),
                "infinite_values": np.isinf(features).sum(),
                "target_range": (targets.min(), targets.max()),
                "target_mean": targets.mean(),
                "target_std": targets.std(),
                "feature_ranges": {
                    f"feature_{i}": (features[:, i].min(), features[:, i].max())
                    for i in range(min(5, features.shape[1]))  # Check first 5
                },
            }

            quality_checks[split_name] = checks

        return quality_checks

    def _generate_stress_scenarios(self, bonds: List[Bond]) -> Dict:
        """
        Generate stress test scenarios

        Industry Best Practice: Stress testing for model validation
        """
        stress_scenarios = {}

        # Scenario 1: Interest rate shock (+200 bps)
        stress_scenarios["rate_shock_up"] = self._apply_stress_scenario(bonds, risk_free_rate_change=0.02)

        # Scenario 2: Interest rate shock (-200 bps)
        stress_scenarios["rate_shock_down"] = self._apply_stress_scenario(bonds, risk_free_rate_change=-0.02)

        # Scenario 3: Credit spread widening (+100 bps)
        stress_scenarios["spread_widening"] = self._apply_stress_scenario(bonds, spread_change=0.01)

        # Scenario 4: Liquidity crisis
        stress_scenarios["liquidity_crisis"] = self._apply_stress_scenario(bonds, liquidity_multiplier=0.3)

        return stress_scenarios

    def _apply_stress_scenario(
        self,
        bonds: List[Bond],
        risk_free_rate_change: float = 0.0,
        spread_change: float = 0.0,
        liquidity_multiplier: float = 1.0,
    ) -> List[Dict]:
        """Apply stress scenario to bonds"""
        scenario_data = []

        original_rf = self.valuator.risk_free_rate
        self.valuator.risk_free_rate = original_rf + risk_free_rate_change

        for bond in bonds[:100]:  # Sample for stress tests
            base_spread = self.valuator._get_credit_spread(bond.credit_rating)
            adjusted_spread = base_spread + spread_change

            required_ytm = self.valuator.risk_free_rate + adjusted_spread
            fair_value = self.valuator.calculate_fair_value(bond, required_yield=required_ytm)

            # Apply liquidity impact
            liquidity_impact = (1.0 - liquidity_multiplier) * 0.05
            market_price = fair_value * (1 - liquidity_impact)

            scenario_data.append(
                {
                    "bond": bond,
                    "fair_value": fair_value,
                    "market_price": market_price,
                    "stress_adjustment": market_price / bond.current_price if bond.current_price > 0 else 1.0,
                }
            )

        self.valuator.risk_free_rate = original_rf

        return scenario_data

    def generate_bonds_for_training(self, num_bonds: int = 1000, include_regimes: List[str] = None) -> List[Bond]:
        """
        Generate bonds specifically formatted for model training

        Convenience method that returns bonds ready for model.train() methods
        """
        if include_regimes is None:
            include_regimes = ["normal", "bull", "bear"]

        bonds = self._generate_diverse_bond_universe(num_bonds)

        # Apply regime effects
        training_bonds = []
        for bond in bonds:
            # Randomly assign regime
            regime_name = np.random.choice(include_regimes)
            regime = self.regimes[regime_name]

            # Update valuator
            self.valuator.risk_free_rate = regime.risk_free_rate

            # Calculate fair value
            base_spread = self.valuator._get_credit_spread(bond.credit_rating)
            adjusted_spread = base_spread + regime.credit_spread_base
            required_ytm = regime.risk_free_rate + adjusted_spread
            fair_value = self.valuator.calculate_fair_value(bond, required_yield=required_ytm)

            # Add market noise
            volatility = 0.02 * regime.volatility_multiplier
            noise = np.random.normal(0, volatility)
            sentiment_impact = regime.market_sentiment * 0.01

            market_price = fair_value * (1 + noise + sentiment_impact)
            market_price = np.clip(market_price, fair_value * 0.5, fair_value * 1.5)

            bond.current_price = market_price
            training_bonds.append(bond)

        return training_bonds


def save_training_dataset(dataset: Dict, filepath: str) -> None:
    """
    Save training dataset to disk

    Args:
        dataset: Training dataset dictionary
        filepath: Path to save dataset
    """
    import joblib

    os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else ".", exist_ok=True)
    joblib.dump(dataset, filepath)
    print(f"Dataset saved to {filepath}")


def load_training_dataset(filepath: str) -> Dict:
    """Load training dataset from disk"""
    import joblib

    return joblib.load(filepath)


if __name__ == "__main__":
    # Example usage
    generator = TrainingDataGenerator(seed=42)

    print("=" * 60)
    print("COMPREHENSIVE TRAINING DATASET GENERATOR")
    print("Following Financial Industry Best Practices")
    print("=" * 60)
    print()

    # Generate dataset
    dataset = generator.generate_comprehensive_dataset(total_bonds=5000, time_periods=60, bonds_per_period=100)

    # Print summary
    print("\n" + "=" * 60)
    print("DATASET SUMMARY")
    print("=" * 60)
    print(f"Total observations: {dataset['dataset_metadata']['total_observations']}")
    print(f"Train size: {dataset['dataset_metadata']['train_size']}")
    print(f"Validation size: {dataset['dataset_metadata']['validation_size']}")
    print(f"Test size: {dataset['dataset_metadata']['test_size']}")
    print(f"Number of features: {dataset['dataset_metadata']['num_features']}")
    print(f"Regimes represented: {', '.join(dataset['dataset_metadata']['regimes_represented'])}")
    date_start = dataset["dataset_metadata"]["date_range"]["start"]
    date_end = dataset["dataset_metadata"]["date_range"]["end"]
    print(f"Date range: {date_start} to {date_end}")

    print("\n" + "=" * 60)
    print("DATA QUALITY REPORT")
    print("=" * 60)
    for split_name, quality in dataset["quality_report"].items():
        print(f"\n{split_name.upper()}:")
        print(f"  Samples: {quality['n_samples']}")
        print(f"  Missing values: {quality['missing_values']}")
        print(f"  Target range: [{quality['target_range'][0]:.3f}, {quality['target_range'][1]:.3f}]")
        print(f"  Target mean: {quality['target_mean']:.3f}")

    # Save dataset
    import os

    save_training_dataset(dataset, "training_data/training_dataset.joblib")
    print("\n✓ Dataset saved successfully!")
