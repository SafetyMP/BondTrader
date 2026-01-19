"""
Unit tests for validation utilities
"""

import os
import pytest
from pathlib import Path
from typing import List

from bondtrader.core.bond_models import Bond, BondType
from bondtrader.utils.validation import (
    validate_bond_input,
    validate_credit_rating,
    validate_file_path,
    validate_list_not_empty,
    validate_numeric_range,
    validate_percentage,
    validate_positive,
    validate_probability,
    validate_weights_sum,
)


class TestNumericValidation:
    """Test numeric validation functions"""

    def test_validate_positive_with_positive_value(self):
        """Test validate_positive with positive value"""
        validate_positive(5.0, name="value")
        validate_positive(0.1, name="value")
        # Should not raise

    def test_validate_positive_with_zero(self):
        """Test validate_positive with zero"""
        with pytest.raises(ValueError, match="must be positive"):
            validate_positive(0.0, name="value")

    def test_validate_positive_with_negative(self):
        """Test validate_positive with negative value"""
        with pytest.raises(ValueError, match="must be positive"):
            validate_positive(-5.0, name="value")

    def test_validate_positive_with_non_numeric(self):
        """Test validate_positive with non-numeric value"""
        with pytest.raises(TypeError, match="must be numeric"):
            validate_positive("5", name="value")

    def test_validate_numeric_range_with_valid_range(self):
        """Test validate_numeric_range with valid range"""
        validate_numeric_range(5.0, min_val=0.0, max_val=10.0, name="value")
        validate_numeric_range(0.0, min_val=0.0, max_val=10.0, name="value")  # Min inclusive
        validate_numeric_range(10.0, min_val=0.0, max_val=10.0, name="value")  # Max inclusive

    def test_validate_numeric_range_below_min(self):
        """Test validate_numeric_range below minimum"""
        with pytest.raises(ValueError, match="must be >= 0"):
            validate_numeric_range(-1.0, min_val=0.0, name="value")

    def test_validate_numeric_range_above_max(self):
        """Test validate_numeric_range above maximum"""
        with pytest.raises(ValueError, match="must be <= 10"):
            validate_numeric_range(11.0, max_val=10.0, name="value")

    def test_validate_percentage_with_valid_percentage(self):
        """Test validate_percentage with valid percentage"""
        validate_percentage(50.0, name="percentage")
        validate_percentage(0.0, name="percentage")
        validate_percentage(100.0, name="percentage")

    def test_validate_percentage_out_of_range(self):
        """Test validate_percentage out of range"""
        with pytest.raises(ValueError):
            validate_percentage(-1.0, name="percentage")
        with pytest.raises(ValueError):
            validate_percentage(101.0, name="percentage")

    def test_validate_probability_with_valid_probability(self):
        """Test validate_probability with valid probability"""
        validate_probability(0.5, name="probability")
        validate_probability(0.0, name="probability")
        validate_probability(1.0, name="probability")

    def test_validate_probability_out_of_range(self):
        """Test validate_probability out of range"""
        with pytest.raises(ValueError):
            validate_probability(-0.1, name="probability")
        with pytest.raises(ValueError):
            validate_probability(1.1, name="probability")


class TestListValidation:
    """Test list validation functions"""

    def test_validate_list_not_empty_with_non_empty_list(self):
        """Test validate_list_not_empty with non-empty list"""
        validate_list_not_empty([1, 2, 3], name="list")
        validate_list_not_empty(["a"], name="list")

    def test_validate_list_not_empty_with_empty_list(self):
        """Test validate_list_not_empty with empty list"""
        with pytest.raises(ValueError, match="cannot be empty"):
            validate_list_not_empty([], name="list")

    def test_validate_list_not_empty_with_non_list(self):
        """Test validate_list_not_empty with non-list"""
        with pytest.raises(TypeError, match="must be a list"):
            validate_list_not_empty("not a list", name="list")
        with pytest.raises(TypeError, match="must be a list"):
            validate_list_not_empty(123, name="list")

    def test_validate_weights_sum_with_valid_weights(self):
        """Test validate_weights_sum with valid weights"""
        validate_weights_sum([0.3, 0.3, 0.4], expected_sum=1.0, name="weights")
        validate_weights_sum([0.5, 0.5], expected_sum=1.0, name="weights")

    def test_validate_weights_sum_not_summing_to_one(self):
        """Test validate_weights_sum with weights not summing to expected"""
        with pytest.raises(ValueError, match="must sum to 1.0"):
            validate_weights_sum([0.3, 0.3], expected_sum=1.0, name="weights")

    def test_validate_weights_sum_with_empty_list(self):
        """Test validate_weights_sum with empty list"""
        with pytest.raises(ValueError, match="cannot be empty"):
            validate_weights_sum([], name="weights")


class TestFilePathValidation:
    """Test file path validation"""

    def test_validate_file_path_with_valid_path(self):
        """Test validate_file_path with valid path"""
        validate_file_path("test_file.txt", name="filepath")
        validate_file_path("/path/to/file.txt", name="filepath")

    def test_validate_file_path_with_empty_path(self):
        """Test validate_file_path with empty path"""
        with pytest.raises(ValueError, match="cannot be empty"):
            validate_file_path("", name="filepath")
        with pytest.raises(ValueError, match="cannot be empty"):
            validate_file_path("   ", name="filepath")

    def test_validate_file_path_with_non_string(self):
        """Test validate_file_path with non-string"""
        with pytest.raises(TypeError, match="must be a string"):
            validate_file_path(123, name="filepath")

    def test_validate_file_path_must_exist(self, tmp_path):
        """Test validate_file_path with must_exist=True"""
        # Create a temp file
        test_file = tmp_path / "test_file.txt"
        test_file.write_text("test")
        
        # Should not raise
        validate_file_path(str(test_file), must_exist=True, name="filepath")

    def test_validate_file_path_must_exist_file_not_found(self):
        """Test validate_file_path with must_exist=True but file doesn't exist"""
        with pytest.raises(FileNotFoundError):
            validate_file_path("nonexistent_file.txt", must_exist=True, name="filepath")


class TestCreditRatingValidation:
    """Test credit rating validation"""

    def test_validate_credit_rating_with_valid_ratings(self):
        """Test validate_credit_rating with valid ratings"""
        validate_credit_rating("AAA", name="credit_rating")
        validate_credit_rating("BBB", name="credit_rating")
        validate_credit_rating("aa", name="credit_rating")  # Case insensitive
        validate_credit_rating("D", name="credit_rating")

    def test_validate_credit_rating_with_invalid_format(self):
        """Test validate_credit_rating with invalid format"""
        # Should warn but not raise for non-standard ratings
        validate_credit_rating("INVALID", name="credit_rating")  # Warning only

    def test_validate_credit_rating_with_non_string(self):
        """Test validate_credit_rating with non-string"""
        with pytest.raises(TypeError, match="must be a string"):
            validate_credit_rating(123, name="credit_rating")


class TestBondInputValidation:
    """Test bond input validation decorator"""

    @validate_bond_input
    def dummy_function_with_bond(self, bond: Bond):
        """Dummy function for testing decorator"""
        return bond.bond_id

    def test_validate_bond_input_with_valid_bond(self):
        """Test validate_bond_input decorator with valid bond"""
        from tests.fixtures.bond_factory import create_test_bond

        bond = create_test_bond()
        result = self.dummy_function_with_bond(bond)
        assert result == bond.bond_id

    def test_validate_bond_input_with_invalid_price(self):
        """Test validate_bond_input decorator with invalid price"""
        from tests.fixtures.bond_factory import create_test_bond

        bond = create_test_bond(current_price=-100)  # Invalid price
        with pytest.raises(ValueError, match="must be positive"):
            self.dummy_function_with_bond(bond)

    def test_validate_bond_input_with_invalid_face_value(self):
        """Test validate_bond_input decorator with invalid face value"""
        from tests.fixtures.bond_factory import create_test_bond

        bond = create_test_bond(face_value=0)  # Invalid face value
        with pytest.raises(ValueError, match="must be positive"):
            self.dummy_function_with_bond(bond)

    def test_validate_bond_input_with_non_bond(self):
        """Test validate_bond_input decorator with non-Bond object"""
        with pytest.raises(TypeError, match="Expected Bond instance"):
            self.dummy_function_with_bond("not a bond")


@pytest.mark.unit
class TestValidationIntegration:
    """Integration tests for validation utilities"""

    def test_combined_validations(self):
        """Test combining multiple validations"""
        weights = [0.3, 0.3, 0.4]
        validate_list_not_empty(weights, name="weights")
        validate_weights_sum(weights, expected_sum=1.0, name="weights")
        for w in weights:
            validate_probability(w, name="weight")

    def test_validation_error_messages(self):
        """Test that validation error messages are informative"""
        with pytest.raises(ValueError) as exc_info:
            validate_positive(-5.0, name="test_value")
        assert "test_value" in str(exc_info.value)
        assert "must be positive" in str(exc_info.value)
