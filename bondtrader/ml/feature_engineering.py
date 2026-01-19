"""
Centralized Feature Engineering for Bond ML Models
Extracts common feature creation logic to eliminate duplication
"""

from datetime import datetime
from typing import List, Tuple

import numpy as np

from bondtrader.core.bond_models import Bond
from bondtrader.core.bond_valuation import BondValuator


class BondFeatureEngineer:
    """Centralized feature engineering for bond ML models"""

    @staticmethod
    def create_basic_features(bonds: List[Bond], fair_values: List[float], valuator: BondValuator) -> np.ndarray:
        """
        Create basic feature set (12 features)

        Features:
        - coupon_rate, time_to_maturity, credit_rating_numeric
        - price_to_par_ratio, years_since_issue, frequency
        - callable, convertible
        - ytm, duration, convexity, face_value
        """
        current_time = datetime.now()
        ytms = [valuator.calculate_yield_to_maturity(bond) for bond in bonds]
        durations = [valuator.calculate_duration(bond, ytm) for bond, ytm in zip(bonds, ytms)]
        convexities = [valuator.calculate_convexity(bond, ytm) for bond, ytm in zip(bonds, ytms)]

        features = []
        for bond, fv, ytm, duration, convexity in zip(bonds, fair_values, ytms, durations, convexities):
            char = bond.get_bond_characteristics(current_time=current_time)

            feature_vector = [
                char["coupon_rate"],
                char["time_to_maturity"],
                char["credit_rating_numeric"],
                char["current_price"] / char["face_value"],
                char["years_since_issue"],
                char["frequency"],
                char["callable"],
                char["convertible"],
                ytm * 100,  # YTM as percentage
                duration,
                convexity,
                bond.face_value,  # Size factor
            ]

            features.append(feature_vector)

        return np.array(features)

    @staticmethod
    def create_enhanced_features(
        bonds: List[Bond], fair_values: List[float], valuator: BondValuator
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Create enhanced feature set (18 features)

        Includes basic features plus:
        - modified_duration, spread_over_rf, time_decay
        - quarter, day_of_year
        """
        feature_names = [
            "coupon_rate",
            "time_to_maturity",
            "credit_rating_numeric",
            "price_to_par_ratio",
            "years_since_issue",
            "frequency",
            "callable",
            "convertible",
            "ytm",
            "duration",
            "convexity",
            "face_value",
            "modified_duration",
            "spread_over_rf",
            "time_decay",
            "quarter",
            "day_of_year",
        ]

        current_date = datetime.now()
        ytms = [valuator.calculate_yield_to_maturity(bond) for bond in bonds]
        durations = [valuator.calculate_duration(bond, ytm) for bond, ytm in zip(bonds, ytms)]
        convexities = [valuator.calculate_convexity(bond, ytm) for bond, ytm in zip(bonds, ytms)]

        features = []
        for bond, fv, ytm, duration, convexity in zip(bonds, fair_values, ytms, durations, convexities):
            char = bond.get_bond_characteristics()

            # Base features (reuse basic feature creation)
            feature_vector = [
                char["coupon_rate"],
                char["time_to_maturity"],
                char["credit_rating_numeric"],
                char["current_price"] / char["face_value"],
                char["years_since_issue"],
                char["frequency"],
                char["callable"],
                char["convertible"],
                ytm * 100,
                duration,
                convexity,
                bond.face_value,
            ]

            # Enhanced features
            modified_duration = duration / (1 + ytm) if ytm > 0 else duration
            spread_over_rf = ytm - valuator.risk_free_rate
            time_decay = (
                char["time_to_maturity"] / (char["years_since_issue"] + char["time_to_maturity"])
                if (char["years_since_issue"] + char["time_to_maturity"]) > 0
                else 0
            )
            quarter = current_date.month // 4 + 1
            day_of_year = current_date.timetuple().tm_yday

            feature_vector.extend([modified_duration, spread_over_rf * 100, time_decay, quarter, day_of_year / 365.25])

            features.append(feature_vector)

        return np.array(features), feature_names

    @staticmethod
    def create_advanced_features(
        bonds: List[Bond], fair_values: List[float], valuator: BondValuator
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Create advanced feature set (30+ features with polynomials)

        Includes enhanced features plus:
        - Polynomial features (degree 2)
        - Interaction features
        """
        ytms = [valuator.calculate_yield_to_maturity(bond) for bond in bonds]
        durations = [valuator.calculate_duration(bond, ytm) for bond, ytm in zip(bonds, ytms)]
        convexities = [valuator.calculate_convexity(bond, ytm) for bond, ytm in zip(bonds, ytms)]

        features = []
        for bond, fv, ytm, duration, convexity in zip(bonds, fair_values, ytms, durations, convexities):
            char = bond.get_bond_characteristics()

            # Base features
            coupon_rate = char["coupon_rate"]
            ttm = char["time_to_maturity"]
            rating_num = char["credit_rating_numeric"]
            price_to_par = char["current_price"] / char["face_value"]
            years_issue = char["years_since_issue"]
            freq = char["frequency"]
            callable_flag = char["callable"]
            convertible_flag = char["convertible"]

            feature_vector = [
                coupon_rate,
                ttm,
                rating_num,
                price_to_par,
                years_issue,
                freq,
                callable_flag,
                convertible_flag,
                ytm * 100,
                duration,
                convexity,
                bond.face_value,
                duration / (1 + ytm) if ytm > 0 else duration,  # Modified duration
                ytm - valuator.risk_free_rate,  # Spread over RF
            ]

            # Polynomial features (degree 2)
            feature_vector.extend(
                [coupon_rate**2, ttm**2, duration**2, coupon_rate * ttm, coupon_rate * duration, ttm * duration]
            )

            # Interaction features
            feature_vector.extend([price_to_par * duration, rating_num * ttm, ytm * duration, convexity * duration])

            features.append(feature_vector)

        # Feature names
        base_names = [
            "coupon_rate",
            "time_to_maturity",
            "credit_rating",
            "price_to_par",
            "years_since_issue",
            "frequency",
            "callable",
            "convertible",
            "ytm",
            "duration",
            "convexity",
            "face_value",
            "modified_duration",
            "spread_over_rf",
        ]
        poly_names = ["coupon_rate^2", "ttm^2", "duration^2", "coupon*ttm", "coupon*duration", "ttm*duration"]
        inter_names = ["price_to_par*duration", "rating*ttm", "ytm*duration", "convexity*duration"]
        feature_names = base_names + poly_names + inter_names

        return np.array(features), feature_names

    @staticmethod
    def create_targets(bonds: List[Bond], fair_values: List[float]) -> np.ndarray:
        """
        Create target values for training
        Target is the adjustment factor: actual_price / theoretical_fair_value
        """
        targets = []
        for bond, fv in zip(bonds, fair_values):
            if fv > 0:
                adjustment = bond.current_price / fv
                targets.append(adjustment)
            else:
                targets.append(1.0)
        return np.array(targets)
