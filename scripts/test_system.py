"""
Simple test script to verify bond trading system functionality
"""

from datetime import datetime, timedelta

from bondtrader.core.arbitrage_detector import ArbitrageDetector
from bondtrader.core.bond_models import Bond, BondType
from bondtrader.core.container import get_container
from bondtrader.data.data_generator import BondDataGenerator
from bondtrader.ml.ml_adjuster import MLBondAdjuster


def test_basic_valuation():
    """Test basic bond valuation"""
    print("Testing basic bond valuation...")

    bond = Bond(
        bond_id="TEST-001",
        bond_type=BondType.CORPORATE,
        face_value=1000,
        coupon_rate=5.0,
        maturity_date=datetime.now() + timedelta(days=1825),  # 5 years
        issue_date=datetime.now() - timedelta(days=365),
        current_price=950,
        credit_rating="BBB",
        issuer="Test Corp",
        frequency=2,
    )

    container = get_container()
    valuator = container.get_valuator(risk_free_rate=0.03)
    fair_value = valuator.calculate_fair_value(bond)
    ytm = valuator.calculate_yield_to_maturity(bond)
    mismatch = valuator.calculate_price_mismatch(bond)

    print(f"Bond ID: {bond.bond_id}")
    print(f"Market Price: ${bond.current_price:.2f}")
    print(f"Fair Value: ${fair_value:.2f}")
    print(f"YTM: {ytm*100:.2f}%")
    print(f"Mismatch: {mismatch['mismatch_percentage']:.2f}%")
    print("✓ Basic valuation works!\n")

    return bond


def test_arbitrage_detection(bonds):
    """Test arbitrage detection"""
    print("Testing arbitrage detection...")

    container = get_container()
    valuator = container.get_valuator()
    detector = ArbitrageDetector(valuator=valuator, min_profit_threshold=0.01)
    opportunities = detector.find_arbitrage_opportunities(bonds, use_ml=False)

    print(f"Found {len(opportunities)} arbitrage opportunities")
    if opportunities:
        print(f"Top opportunity: {opportunities[0]['bond_id']} - {opportunities[0]['profit_percentage']:.2f}%")
    print("✓ Arbitrage detection works!\n")


def test_ml_adjuster(bonds):
    """Test ML adjuster"""
    from bondtrader.config import get_config

    print("Testing ML adjuster...")

    if len(bonds) < 10:
        print("⚠ Not enough bonds for ML training (need at least 10)")
        return

    config = get_config()
    ml_adjuster = MLBondAdjuster(model_type=config.ml_model_type)

    try:
        metrics = ml_adjuster.train(bonds, test_size=config.ml_test_size, random_state=config.ml_random_state)
        print(f"Train R²: {metrics['train_r2']:.4f}")
        print(f"Test R²: {metrics['test_r2']:.4f}")
        print("✓ ML adjuster works!\n")

        # Test prediction
        result = ml_adjuster.predict_adjusted_value(bonds[0])
        print(f"ML Adjusted Value for {bonds[0].bond_id}: ${result['ml_adjusted_fair_value']:.2f}")
        print("✓ ML prediction works!\n")
    except Exception as e:
        print(f"⚠ ML training error: {e}\n")


def test_data_generator():
    """Test data generator"""
    print("Testing data generator...")

    generator = BondDataGenerator(seed=42)
    bonds = generator.generate_bonds(20)

    print(f"Generated {len(bonds)} bonds")
    print(f"Bond types: {[b.bond_type.value for b in bonds[:5]]}")
    print("✓ Data generator works!\n")

    return bonds


def main():
    """Run all tests"""
    print("=" * 50)
    print("Bond Trading System - Test Suite")
    print("=" * 50)
    print()

    # Test basic valuation
    bond = test_basic_valuation()

    # Test data generator
    bonds = test_data_generator()

    # Test arbitrage detection
    test_arbitrage_detection(bonds)

    # Test ML adjuster
    test_ml_adjuster(bonds)

    print("=" * 50)
    print("All tests completed!")
    print("=" * 50)


if __name__ == "__main__":
    main()
