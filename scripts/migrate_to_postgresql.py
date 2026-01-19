"""
PostgreSQL Migration Script
Migrates from SQLite to PostgreSQL or creates PostgreSQL schema

CRITICAL: Run this script to set up PostgreSQL database
"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from bondtrader.data.data_persistence import Base, EnhancedBondDatabase
from bondtrader.data.postgresql_support import PostgreSQLDatabase, get_database_type
from bondtrader.utils.utils import logger


def migrate_schema_to_postgresql():
    """
    Create PostgreSQL schema from SQLAlchemy models.

    This creates all tables, constraints, and indexes.
    """
    db_type = get_database_type()

    if db_type != "postgresql":
        logger.warning(
            f"Database type is {db_type}, not PostgreSQL. Set DATABASE_TYPE=postgresql to migrate."
        )
        logger.info("Creating PostgreSQL schema anyway for testing...")

    try:
        # Create PostgreSQL database instance
        pg_db = PostgreSQLDatabase()

        # Create all tables from SQLAlchemy Base
        logger.info("Creating PostgreSQL schema...")
        pg_db.create_tables(Base)

        logger.info("✅ PostgreSQL schema created successfully!")
        logger.info("Tables created: bonds, price_history, valuations, arbitrage_opportunities")

        return True
    except Exception as e:
        logger.error(f"❌ Failed to create PostgreSQL schema: {e}")
        return False


def migrate_data_from_sqlite(sqlite_path: str = "bonds.db"):
    """
    Migrate data from SQLite to PostgreSQL.

    Args:
        sqlite_path: Path to SQLite database file
    """
    try:
        # Load data from SQLite
        logger.info(f"Loading data from SQLite: {sqlite_path}")
        sqlite_db = EnhancedBondDatabase(db_path=sqlite_path)
        bonds = sqlite_db.load_all_bonds()

        if not bonds:
            logger.info("No bonds found in SQLite database. Skipping data migration.")
            return True

        logger.info(f"Found {len(bonds)} bonds to migrate")

        # Save to PostgreSQL
        pg_db = PostgreSQLDatabase()

        logger.info("Migrating bonds to PostgreSQL...")
        saved_count = 0
        for bond in bonds:
            try:
                # Use the same save_bond method (it will work with PostgreSQL)
                # We need to use the PostgreSQL database's save method
                # For now, we'll create a session and save manually
                session = pg_db.get_session()
                try:
                    from datetime import datetime

                    from bondtrader.data.data_persistence import BondModel

                    # Check if bond exists
                    existing = (
                        session.query(BondModel).filter(BondModel.bond_id == bond.bond_id).first()
                    )
                    if existing:
                        logger.debug(f"Bond {bond.bond_id} already exists, skipping")
                        continue

                    # Create new bond model
                    bond_model = BondModel(
                        bond_id=bond.bond_id,
                        bond_type=bond.bond_type.value,
                        face_value=bond.face_value,
                        coupon_rate=bond.coupon_rate,
                        maturity_date=bond.maturity_date.isoformat(),
                        issue_date=bond.issue_date.isoformat(),
                        current_price=bond.current_price,
                        credit_rating=bond.credit_rating,
                        issuer=bond.issuer,
                        frequency=bond.frequency,
                        callable=bond.callable,
                        convertible=bond.convertible,
                        updated_at=datetime.now().isoformat(),
                    )
                    session.add(bond_model)
                    session.commit()
                    saved_count += 1

                    if saved_count % 100 == 0:
                        logger.info(f"Migrated {saved_count}/{len(bonds)} bonds...")
                finally:
                    session.close()
            except Exception as e:
                logger.error(f"Error migrating bond {bond.bond_id}: {e}")
                continue

        logger.info(f"✅ Successfully migrated {saved_count}/{len(bonds)} bonds to PostgreSQL!")
        return True
    except Exception as e:
        logger.error(f"❌ Failed to migrate data: {e}")
        return False


def main():
    """Main migration function"""
    import argparse

    parser = argparse.ArgumentParser(description="Migrate to PostgreSQL")
    parser.add_argument(
        "--schema-only", action="store_true", help="Only create schema, don't migrate data"
    )
    parser.add_argument(
        "--data-only", action="store_true", help="Only migrate data, assume schema exists"
    )
    parser.add_argument("--sqlite-path", default="bonds.db", help="Path to SQLite database file")

    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("PostgreSQL Migration Script")
    logger.info("=" * 60)

    # Check if PostgreSQL is configured
    if get_database_type() != "postgresql":
        logger.warning("⚠️  DATABASE_TYPE is not set to 'postgresql'")
        logger.info("Set environment variables:")
        logger.info("  export DATABASE_TYPE=postgresql")
        logger.info("  export POSTGRES_HOST=localhost")
        logger.info("  export POSTGRES_PORT=5432")
        logger.info("  export POSTGRES_DB=bondtrader")
        logger.info("  export POSTGRES_USER=bondtrader")
        logger.info("  export POSTGRES_PASSWORD=your_password")
        logger.info("")
        logger.info("Continuing with schema creation anyway...")

    # Create schema
    if not args.data_only:
        success = migrate_schema_to_postgresql()
        if not success:
            logger.error("Schema migration failed. Exiting.")
            sys.exit(1)

    # Migrate data
    if not args.schema_only:
        success = migrate_data_from_sqlite(args.sqlite_path)
        if not success:
            logger.error("Data migration failed.")
            sys.exit(1)

    logger.info("=" * 60)
    logger.info("✅ Migration complete!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
