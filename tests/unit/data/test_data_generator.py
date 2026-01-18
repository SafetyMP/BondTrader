"""
Unit tests for bond data generator module
"""

import os
import sys

import pytest

pytestmark = pytest.mark.unit

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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
