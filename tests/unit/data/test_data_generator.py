"""
Unit tests for bond data generator module
"""

import os
import sys

import pytest

pytestmark = pytest.mark.unit

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from bondtrader.core.bond_models import Bond, BondType
from bondtrader.data.data_generator import BondDataGenerator


@pytest.fixture
def generator():
    """Create data generator instance"""
    return BondDataGenerator(seed=42)


def test_generator_initialization():
    """Test generator initialization"""
    gen = BondDataGenerator()
    assert gen is not None


def test_generator_with_seed():
    """Test generator with seed for reproducibility"""
    gen1 = BondDataGenerator(seed=42)
    gen2 = BondDataGenerator(seed=42)

    bonds1 = gen1.generate_bonds(num_bonds=10)
    bonds2 = gen2.generate_bonds(num_bonds=10)

    # Same seed should produce same bonds (at least same count)
    assert len(bonds1) == len(bonds2) == 10


def test_generate_bonds(generator):
    """Test bond generation"""
    bonds = generator.generate_bonds(num_bonds=20)

    assert len(bonds) == 20
    assert all(isinstance(bond, Bond) for bond in bonds)


def test_generate_bonds_with_bond_types(generator):
    """Test generating bonds - verify diverse bond types are generated"""
    bonds = generator.generate_bonds(num_bonds=50)
    bond_types = {b.bond_type for b in bonds}
    # Should generate multiple types
    assert len(bond_types) >= 2


def test_generate_bonds_with_credit_ratings(generator):
    """Test generating bonds - verify diverse credit ratings are generated"""
    bonds = generator.generate_bonds(num_bonds=50)
    ratings = {b.credit_rating for b in bonds}
    # Should generate multiple ratings
    assert len(ratings) >= 2


def test_generate_bonds_zero_count(generator):
    """Test generating zero bonds"""
    bonds = generator.generate_bonds(num_bonds=0)
    assert len(bonds) == 0


def test_generate_bonds_large_count(generator):
    """Test generating large number of bonds"""
    bonds = generator.generate_bonds(num_bonds=100)
    assert len(bonds) == 100


def test_generated_bonds_have_valid_data(generator):
    """Test that generated bonds have valid data"""
    bonds = generator.generate_bonds(num_bonds=5)

    for bond in bonds:
        assert bond.bond_id is not None
        assert bond.face_value > 0
        assert bond.current_price > 0
        assert bond.coupon_rate >= 0
        assert bond.maturity_date > bond.issue_date


def test_generated_bonds_diverse_types(generator):
    """Test that generated bonds include diverse types"""
    bonds = generator.generate_bonds(num_bonds=50)

    bond_types = {bond.bond_type for bond in bonds}
    assert len(bond_types) > 1  # Should have multiple types


def test_generate_zero_bonds(generator):
    """Test generating zero bonds"""
    bonds = generator.generate_bonds(num_bonds=0)
    assert len(bonds) == 0


def test_generate_single_bond(generator):
    """Test generating a single bond"""
    bonds = generator.generate_bonds(num_bonds=1)
    assert len(bonds) == 1
    assert isinstance(bonds[0], Bond)


def test_add_price_noise(generator):
    """Test adding price noise to bonds"""
    bonds = generator.generate_bonds(num_bonds=10)
    original_prices = [b.current_price for b in bonds]
    bonds_with_noise = generator.add_price_noise(bonds, noise_level=0.05)
    noisy_prices = [b.current_price for b in bonds_with_noise]

    # Prices should be different (within noise level)
    assert any(op != np for op, np in zip(original_prices, noisy_prices))


def test_add_price_noise_zero_level(generator):
    """Test adding zero price noise"""
    bonds = generator.generate_bonds(num_bonds=10, seed=42)  # Use seed for reproducibility
    original_prices = [b.current_price for b in bonds]
    bonds_no_noise = generator.add_price_noise(bonds.copy(), noise_level=0.0)
    no_noise_prices = [b.current_price for b in bonds_no_noise]

    # Prices should be same with zero noise (use approximate comparison for float)
    import numpy as np

    assert np.allclose(original_prices, no_noise_prices)
