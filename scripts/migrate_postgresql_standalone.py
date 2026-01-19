#!/usr/bin/env python3
"""
Standalone PostgreSQL Migration Script
Creates PostgreSQL schema without importing full bondtrader module

This avoids XGBoost dependency issues during migration.
"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def create_postgresql_schema():
    """Create PostgreSQL schema directly using SQLAlchemy"""
    try:
        from sqlalchemy import (
            Boolean,
            CheckConstraint,
            Column,
            Float,
            ForeignKey,
            Integer,
            String,
            Text,
            create_engine,
        )
        from sqlalchemy.orm import declarative_base, sessionmaker

        Base = declarative_base()

        # Define models directly (matching data_persistence.py)
        class BondModel(Base):
            __tablename__ = "bonds"

            bond_id = Column(String, primary_key=True)
            bond_type = Column(String, nullable=False)
            face_value = Column(Float, nullable=False)
            coupon_rate = Column(Float, nullable=False)
            maturity_date = Column(String, nullable=False)
            issue_date = Column(String, nullable=False)
            current_price = Column(Float, nullable=False)
            credit_rating = Column(String)
            issuer = Column(String)
            frequency = Column(Integer, default=2)
            callable = Column(Boolean, default=False)
            convertible = Column(Boolean, default=False)
            created_at = Column(String)
            updated_at = Column(String)

            __table_args__ = (
                CheckConstraint("face_value > 0", name="check_face_value_positive"),
                CheckConstraint("current_price > 0", name="check_price_positive"),
                CheckConstraint(
                    "coupon_rate >= 0 AND coupon_rate <= 1", name="check_coupon_rate_range"
                ),
                CheckConstraint("frequency >= 1 AND frequency <= 12", name="check_frequency_range"),
            )

        class PriceHistoryModel(Base):
            __tablename__ = "price_history"

            id = Column(Integer, primary_key=True, autoincrement=True)
            bond_id = Column(
                String, ForeignKey("bonds.bond_id", ondelete="CASCADE"), nullable=False
            )
            price = Column(Float, nullable=False)
            fair_value = Column(Float)
            timestamp = Column(String, nullable=False)

            __table_args__ = (CheckConstraint("price > 0", name="check_price_history_positive"),)

        class ValuationModel(Base):
            __tablename__ = "valuations"

            id = Column(Integer, primary_key=True, autoincrement=True)
            bond_id = Column(
                String, ForeignKey("bonds.bond_id", ondelete="CASCADE"), nullable=False
            )
            fair_value = Column(Float, nullable=False)
            ytm = Column(Float, nullable=False)
            duration = Column(Float, nullable=False)
            convexity = Column(Float)
            timestamp = Column(String, nullable=False)

            __table_args__ = (
                CheckConstraint("fair_value > 0", name="check_fair_value_positive"),
                CheckConstraint("ytm >= 0", name="check_ytm_positive"),
                CheckConstraint("duration >= 0", name="check_duration_positive"),
            )

        class ArbitrageOpportunityModel(Base):
            __tablename__ = "arbitrage_opportunities"

            id = Column(Integer, primary_key=True, autoincrement=True)
            bond_id = Column(
                String, ForeignKey("bonds.bond_id", ondelete="CASCADE"), nullable=False
            )
            profit_percentage = Column(Float, nullable=False)
            recommendation = Column(String)
            timestamp = Column(String, nullable=False)

            __table_args__ = (
                CheckConstraint(
                    "profit_percentage >= -100 AND profit_percentage <= 1000",
                    name="check_profit_percentage_range",
                ),
            )

        # Get PostgreSQL connection parameters
        host = os.getenv("POSTGRES_HOST", "localhost")
        port = int(os.getenv("POSTGRES_PORT", "5432"))
        database = os.getenv("POSTGRES_DB", "bondtrader")
        user = os.getenv("POSTGRES_USER", "bondtrader")
        password = os.getenv("POSTGRES_PASSWORD", "")

        if not password:
            print("⚠️  POSTGRES_PASSWORD not set. Using empty password.")

        # Build connection string
        connection_string = f"postgresql://{user}:{password}@{host}:{port}/{database}"

        print(f"Connecting to PostgreSQL: {host}:{port}/{database}")

        # Create engine
        engine = create_engine(
            connection_string,
            pool_size=10,
            max_overflow=20,
            pool_pre_ping=True,
            pool_recycle=3600,
            echo=False,
        )

        # Test connection
        from sqlalchemy import text

        with engine.connect() as conn:
            result = conn.execute(text("SELECT 1"))
            print("✅ Connected to PostgreSQL successfully!")

        # Create all tables
        print("Creating PostgreSQL schema...")
        Base.metadata.create_all(bind=engine)

        print("✅ PostgreSQL schema created successfully!")
        print("Tables created:")
        print("  - bonds")
        print("  - price_history")
        print("  - valuations")
        print("  - arbitrage_opportunities")

        return True

    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("Install with: pip install sqlalchemy psycopg2-binary")
        return False
    except Exception as e:
        print(f"❌ Migration failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    """Main function"""
    print("=" * 60)
    print("PostgreSQL Migration (Standalone)")
    print("=" * 60)

    # Check if PostgreSQL is configured
    db_type = os.getenv("DATABASE_TYPE", "sqlite").lower()

    if db_type != "postgresql":
        print("⚠️  DATABASE_TYPE is not set to 'postgresql'")
        print("\nTo run migration, set environment variables:")
        print("  export DATABASE_TYPE=postgresql")
        print("  export POSTGRES_HOST=localhost")
        print("  export POSTGRES_PORT=5432")
        print("  export POSTGRES_DB=bondtrader")
        print("  export POSTGRES_USER=bondtrader")
        print("  export POSTGRES_PASSWORD=your_password")
        print("\nContinuing anyway...")

    success = create_postgresql_schema()

    if success:
        print("\n" + "=" * 60)
        print("✅ Migration complete!")
        print("=" * 60)
        sys.exit(0)
    else:
        print("\n" + "=" * 60)
        print("❌ Migration failed!")
        print("=" * 60)
        sys.exit(1)


if __name__ == "__main__":
    main()
