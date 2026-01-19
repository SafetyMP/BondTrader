"""
Unit tests for regime models
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

from bondtrader.core.bond_models import Bond, BondType
from bondtrader.core.bond_valuation import BondValuator
from bondtrader.ml.regime_models import RegimeDetector


@pytest.mark.unit
class TestRegimeDetector:
    """Test RegimeDetector class"""

    @pytest.fixture
    def valuator(self):
        """Create valuator"""
        return BondValuator(risk_free_rate=0.03)

    @pytest.fixture
    def detector(self, valuator):
        """Create regime detector"""
        return RegimeDetector(valuator=valuator)

    @pytest.fixture
    def sample_bonds(self):
        """Create sample bonds"""
        now = datetime.now()
        return [
            Bond(
                bond_id=f"BOND-{i:03d}",
                bond_type=BondType.CORPORATE,
                face_value=1000,
                coupon_rate=5.0 + i * 0.1,
                maturity_date=now + timedelta(days=1825 + i * 30),
                issue_date=now - timedelta(days=365),
                current_price=950 + i * 5,
                credit_rating="BBB",
                issuer=f"Corp {i}",
                frequency=2,
            )
            for i in range(20)
        ]

    def test_regime_detector_creation(self, detector):
        """Test creating regime detector"""
        assert detector.valuator is not None
        assert detector.regimes is None

    def test_detect_regimes_kmeans(self, detector, sample_bonds):
        """Test detecting regimes with KMeans"""
        result = detector.detect_regimes(sample_bonds, num_regimes=3, method="kmeans")
        assert "regime_labels" in result
        assert "num_regimes" in result
        assert result["num_regimes"] == 3
        assert result["method"] == "kmeans"
        assert len(result["regime_labels"]) == len(sample_bonds)

    def test_detect_regimes_gmm(self, detector, sample_bonds):
        """Test detecting regimes with GMM"""
        result = detector.detect_regimes(sample_bonds, num_regimes=2, method="gmm")
        assert "regime_labels" in result
        assert result["method"] == "gmm"
        assert result["num_regimes"] == 2

    def test_detect_regimes_regime_analysis(self, detector, sample_bonds):
        """Test regime analysis in detection"""
        result = detector.detect_regimes(sample_bonds, num_regimes=3)
        assert "regime_analysis" in result
        assert len(result["regime_analysis"]) > 0

    def test_classify_regime(self, detector):
        """Test regime classification"""
        regime_type = detector._classify_regime(0.05, 0.06, 5.0)  # High spread
        assert "High Stress" in regime_type or "Wide Spreads" in regime_type

        regime_type = detector._classify_regime(0.06, 0.02, 5.0)  # High yield
        assert "High Yield" in regime_type

        regime_type = detector._classify_regime(0.03, 0.02, 8.0)  # Long duration
        assert "Long Duration" in regime_type

    def test_regime_dependent_pricing(self, detector, sample_bonds):
        """Test regime-dependent pricing"""
        # First detect regimes
        detector.detect_regimes(sample_bonds, num_regimes=3)
        
        # Price bond with regime
        result = detector.regime_dependent_pricing(sample_bonds[0], current_regime=0)
        assert "current_regime" in result
        assert "base_fair_value" in result
        assert "regime_adjusted_value" in result
        assert "regime_adjustment" in result

    def test_regime_dependent_pricing_auto_detect(self, detector, sample_bonds):
        """Test regime-dependent pricing with auto detection"""
        detector.detect_regimes(sample_bonds, num_regimes=2)
        
        result = detector.regime_dependent_pricing(sample_bonds[0])
        assert "current_regime" in result
        assert "regime_adjusted_value" in result

    def test_adaptive_risk_metrics(self, detector, sample_bonds):
        """Test adaptive risk metrics"""
        detector.detect_regimes(sample_bonds, num_regimes=2)
        
        result = detector.adaptive_risk_metrics(sample_bonds)
        assert "regime_risks" in result
        assert "num_regimes" in result

    def test_adaptive_risk_metrics_with_weights(self, detector, sample_bonds):
        """Test adaptive risk metrics with weights"""
        detector.detect_regimes(sample_bonds, num_regimes=2)
        weights = [1.0 / len(sample_bonds)] * len(sample_bonds)
        
        result = detector.adaptive_risk_metrics(sample_bonds, weights=weights)
        assert "regime_risks" in result

    def test_regime_dependent_pricing_no_regimes(self, detector, sample_bonds):
        """Test regime-dependent pricing without detected regimes"""
        with pytest.raises(ValueError, match="Regimes not detected"):
            detector.regime_dependent_pricing(sample_bonds[0])

    def test_adaptive_risk_metrics_auto_detect(self, detector, sample_bonds):
        """Test adaptive risk metrics auto-detects regimes"""
        result = detector.adaptive_risk_metrics(sample_bonds)
        assert "regime_risks" in result
        assert detector.regimes is not None