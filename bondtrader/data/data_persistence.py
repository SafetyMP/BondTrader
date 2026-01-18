"""
Data Persistence Module (DEPRECATED)
SQLite database for storing bond data, prices, and ML models

NOTE: This module is deprecated. Use bondtrader.data.data_persistence_enhanced
instead, which provides better performance with SQLAlchemy connection pooling.
The EnhancedBondDatabase class provides the same API and is backward compatible.

This module now simply re-exports from data_persistence_enhanced for backward compatibility.
This file will be removed in a future version - update imports to use data_persistence_enhanced directly.
"""

# Re-export from enhanced module for backward compatibility
from bondtrader.data.data_persistence_enhanced import BondDatabase, EnhancedBondDatabase

__all__ = ["BondDatabase", "EnhancedBondDatabase"]
