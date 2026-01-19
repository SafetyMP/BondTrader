"""
PostgreSQL Migration Support
Provides PostgreSQL database support with migration utilities

CRITICAL: Required for production scale in Fortune 10 financial institutions
"""

import os
from typing import Optional

from bondtrader.utils.utils import logger

# SQLAlchemy imports
try:
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    _sqlalchemy_available = True
except ImportError:
    _sqlalchemy_available = False
    logger.warning("SQLAlchemy not available for PostgreSQL support")


class PostgreSQLDatabase:
    """
    PostgreSQL Database Support

    Provides PostgreSQL connection with same API as SQLite database.
    """

    def __init__(
        self,
        host: Optional[str] = None,
        port: Optional[int] = None,
        database: Optional[str] = None,
        user: Optional[str] = None,
        password: Optional[str] = None,
        connection_string: Optional[str] = None,
        pool_size: int = 10,
        max_overflow: int = 20,
    ):
        """
        Initialize PostgreSQL database connection.

        Args:
            host: Database host (defaults to environment variable)
            port: Database port (defaults to 5432)
            database: Database name (defaults to environment variable)
            user: Database user (defaults to environment variable)
            password: Database password (defaults to environment variable)
            connection_string: Full connection string (overrides other params)
            pool_size: Connection pool size
            max_overflow: Maximum overflow connections
        """
        if not _sqlalchemy_available:
            raise ImportError("SQLAlchemy required for PostgreSQL support")

        # Get connection parameters from environment if not provided
        if connection_string:
            self.connection_string = connection_string
        else:
            host = host or os.getenv("POSTGRES_HOST", "localhost")
            port = port or int(os.getenv("POSTGRES_PORT", "5432"))
            database = database or os.getenv("POSTGRES_DB", "bondtrader")
            user = user or os.getenv("POSTGRES_USER", "bondtrader")
            password = password or os.getenv("POSTGRES_PASSWORD", "")

            # Build connection string
            self.connection_string = f"postgresql://{user}:{password}@{host}:{port}/{database}"

        # Create engine with connection pooling
        self.engine = create_engine(
            self.connection_string,
            pool_size=pool_size,
            max_overflow=max_overflow,
            pool_pre_ping=True,  # Verify connections before use
            pool_recycle=3600,  # Recycle connections after 1 hour
            echo=False,
        )

        # Create session factory
        self.SessionLocal = sessionmaker(bind=self.engine, autocommit=False, autoflush=False)

        logger.info(f"PostgreSQL database initialized: {host}:{port}/{database}")

    def get_session(self):
        """Get database session"""
        return self.SessionLocal()

    def create_tables(self, base):
        """
        Create all tables from SQLAlchemy Base.

        Args:
            base: SQLAlchemy declarative_base() instance
        """
        base.metadata.create_all(bind=self.engine)
        logger.info("PostgreSQL tables created")

    def drop_tables(self, base):
        """
        Drop all tables (use with caution!).

        Args:
            base: SQLAlchemy declarative_base() instance
        """
        base.metadata.drop_all(bind=self.engine)
        logger.warning("PostgreSQL tables dropped")

    def execute_migration(self, migration_sql: str):
        """
        Execute raw SQL migration.

        Args:
            migration_sql: SQL migration script
        """
        with self.engine.connect() as connection:
            connection.execute(migration_sql)
            connection.commit()
        logger.info("Migration executed successfully")


def get_database_type() -> str:
    """
    Get database type from environment.

    Returns:
        "postgresql" or "sqlite"
    """
    return os.getenv("DATABASE_TYPE", "sqlite").lower()


def create_database_instance():
    """
    Create appropriate database instance based on configuration.

    Returns:
        EnhancedBondDatabase or PostgreSQLDatabase instance
    """
    db_type = get_database_type()

    if db_type == "postgresql":
        from bondtrader.data.postgresql_support import PostgreSQLDatabase

        return PostgreSQLDatabase()
    else:
        from bondtrader.data.data_persistence import EnhancedBondDatabase

        db_path = os.getenv("BOND_DB_PATH", "bonds.db")
        return EnhancedBondDatabase(db_path=db_path)
