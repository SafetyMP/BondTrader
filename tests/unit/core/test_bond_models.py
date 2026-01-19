"""
Unit tests for bond models module
"""

import os
import sys
from datetime import datetime, timedelta

import pytest

pytestmark = pytest.mark.unit

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fixtures.bond_factory import create_test_bond

from bondtrader.core.bond_models import Bond, BondClassifier, BondType


def test_bond_creation():
    """Test bond creation with all parameters"""
    bond = Bond(
        bond_id="TEST-001",
        bond_type=BondType.CORPORATE,
        face_value=1000,
        coupon_rate=5.0,
        maturity_date=datetime.now() + timedelta(days=1825),
        issue_date=datetime.now() - timedelta(days=365),
        current_price=950,
        credit_rating="BBB",
        issuer="Test Corp",
    )

    assert bond.bond_id == "TEST-001"
    assert bond.face_value == 1000
    assert bond.coupon_rate == 5.0


def test_bond_get_characteristics():
    """Test bond characteristics extraction"""
    bond = create_test_bond()
    characteristics = bond.get_bond_characteristics()

    assert "coupon_rate" in characteristics
    assert "time_to_maturity" in characteristics
    assert "credit_rating_numeric" in characteristics
    assert "years_since_issue" in characteristics
    assert characteristics["time_to_maturity"] > 0


def test_bond_type_enum():
    """Test bond type enumeration"""
    assert BondType.TREASURY in BondType
    assert BondType.CORPORATE in BondType
    assert BondType.FIXED_RATE in BondType


def test_bond_classifier():
    """Test bond classifier"""
    classifier = BondClassifier()
    bond = create_test_bond()

    classification = classifier.classify(bond)

    assert isinstance(classification, BondType)
    assert classification in BondType


def test_bond_time_to_maturity():
    """Test time to maturity calculation"""
    now = datetime.now()
    bond = create_test_bond()
    bond.maturity_date = now + timedelta(days=365)

    ttm = bond.time_to_maturity
    assert abs(ttm - 1.0) < 0.1  # Approximately 1 year


def test_bond_years_since_issue():
    """Test years since issue calculation"""
    now = datetime.now()
    bond = create_test_bond()
    bond.issue_date = now - timedelta(days=730)

    years = bond.years_since_issue
    assert abs(years - 2.0) < 0.1  # Approximately 2 years


@pytest.mark.unit
class TestBondClassifier:
    """Test BondClassifier functionality"""

    def test_classify_zero_coupon(self):
        """Test classifying zero coupon bond"""
        classifier = BondClassifier()
        bond = Bond(
            bond_id="ZERO-001",
            bond_type=BondType.ZERO_COUPON,
            face_value=1000,
            coupon_rate=0.0,
            maturity_date=datetime.now() + timedelta(days=1825),
            issue_date=datetime.now() - timedelta(days=365),
            current_price=900,
        )
        assert classifier.classify(bond) == BondType.ZERO_COUPON

    def test_classify_treasury(self):
        """Test classifying treasury bond"""
        classifier = BondClassifier()
        bond = Bond(
            bond_id="TREASURY-001",
            bond_type=BondType.TREASURY,
            face_value=1000,
            coupon_rate=3.0,
            maturity_date=datetime.now() + timedelta(days=1825),
            issue_date=datetime.now() - timedelta(days=365),
            current_price=980,
            credit_rating="AAA",
            issuer="US Treasury",
        )
        assert classifier.classify(bond) == BondType.TREASURY

    def test_classify_high_yield(self):
        """Test classifying high yield bond"""
        classifier = BondClassifier()
        bond = Bond(
            bond_id="HY-001",
            bond_type=BondType.HIGH_YIELD,
            face_value=1000,
            coupon_rate=8.0,
            maturity_date=datetime.now() + timedelta(days=1825),
            issue_date=datetime.now() - timedelta(days=365),
            current_price=950,
            credit_rating="BB",
        )
        assert classifier.classify(bond) == BondType.HIGH_YIELD

    def test_extract_features(self):
        """Test extracting features from bonds"""
        classifier = BondClassifier()
        bonds = [
            create_test_bond(bond_id="TEST-001"),
            create_test_bond(bond_id="TEST-002"),
        ]
        features = classifier.extract_features(bonds)
        assert features.shape[0] == 2
        assert features.shape[1] == 8  # 8 features per bond

    def test_extract_features_empty_list(self):
        """Test extracting features from empty list"""
        classifier = BondClassifier()
        features = classifier.extract_features([])
        assert features.shape[0] == 0
