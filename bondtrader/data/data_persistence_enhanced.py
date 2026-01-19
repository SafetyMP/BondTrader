"""
Enhanced Data Persistence Module with SQLAlchemy
Provides connection pooling and better performance than raw SQLite
Maintains backward compatibility with BondDatabase API
"""

import json
import pickle
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from sqlalchemy import (
    Boolean,
    Column,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    create_engine,
    func,
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import Session, sessionmaker

from bondtrader.core.bond_models import Bond, BondType
from bondtrader.utils.utils import logger

# SQLAlchemy Base
Base = declarative_base()


# SQLAlchemy Models
class BondModel(Base):
    """SQLAlchemy model for bonds table"""

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
    created_at = Column(String, default=func.datetime("now"))
    updated_at = Column(String, default=func.datetime("now"))


class PriceHistoryModel(Base):
    """SQLAlchemy model for price_history table"""

    __tablename__ = "price_history"

    id = Column(Integer, primary_key=True, autoincrement=True)
    bond_id = Column(String, ForeignKey("bonds.bond_id"), nullable=False)
    price = Column(Float, nullable=False)
    fair_value = Column(Float)
    timestamp = Column(String, nullable=False)


class ValuationModel(Base):
    """SQLAlchemy model for valuations table"""

    __tablename__ = "valuations"

    id = Column(Integer, primary_key=True, autoincrement=True)
    bond_id = Column(String, ForeignKey("bonds.bond_id"), nullable=False)
    fair_value = Column(Float, nullable=False)
    ytm = Column(Float)
    duration = Column(Float)
    convexity = Column(Float)
    timestamp = Column(String, nullable=False)


class ArbitrageOpportunityModel(Base):
    """SQLAlchemy model for arbitrage_opportunities table"""

    __tablename__ = "arbitrage_opportunities"

    id = Column(Integer, primary_key=True, autoincrement=True)
    bond_id = Column(String, ForeignKey("bonds.bond_id"), nullable=False)
    profit_percentage = Column(Float, nullable=False)
    recommendation = Column(String)
    timestamp = Column(String, nullable=False)


class EnhancedBondDatabase:
    """
    Enhanced SQLite database with SQLAlchemy for connection pooling
    Provides same API as BondDatabase but with better performance
    """

    def __init__(self, db_path: str = "bonds.db", pool_size: int = None):
        """
        Initialize enhanced database with connection pooling

        Args:
            db_path: Path to SQLite database file
            pool_size: Size of connection pool (None = auto-detect based on CPU cores)
        """
        self.db_path = db_path

        # OPTIMIZED: Auto-size connection pool based on system resources
        if pool_size is None:
            import multiprocessing

            # SQLite works best with small pools, but we allow more for concurrent reads
            # Default to min(10, CPU cores + 2) for better concurrency
            pool_size = min(10, multiprocessing.cpu_count() + 2)

        # SQLite connection string with connection pooling
        # Use check_same_thread=False for connection pooling
        # OPTIMIZED: Added pool_recycle to prevent stale connections
        engine_url = f"sqlite:///{db_path}?check_same_thread=False"
        self.engine = create_engine(
            engine_url,
            pool_size=pool_size,
            max_overflow=pool_size * 2,  # Allow overflow for burst traffic
            pool_pre_ping=True,  # Verify connections before use
            pool_recycle=3600,  # Recycle connections after 1 hour
            echo=False,
        )
        self.SessionLocal = sessionmaker(bind=self.engine, autocommit=False, autoflush=False)
        self._init_database()

    def _init_database(self):
        """Initialize database tables using SQLAlchemy"""
        Base.metadata.create_all(bind=self.engine)

    def _get_session(self) -> Session:
        """Get database session (use in context manager)"""
        return self.SessionLocal()

    def save_bond(self, bond: Bond) -> bool:
        """Save bond to database"""
        session = self._get_session()
        try:
            # Check if bond exists
            existing = session.query(BondModel).filter(BondModel.bond_id == bond.bond_id).first()
            if existing:
                # Update existing
                existing.bond_type = bond.bond_type.value
                existing.face_value = bond.face_value
                existing.coupon_rate = bond.coupon_rate
                existing.maturity_date = bond.maturity_date.isoformat()
                existing.issue_date = bond.issue_date.isoformat()
                existing.current_price = bond.current_price
                existing.credit_rating = bond.credit_rating
                existing.issuer = bond.issuer
                existing.frequency = bond.frequency
                existing.callable = bond.callable
                existing.convertible = bond.convertible
                existing.updated_at = datetime.now().isoformat()
            else:
                # Insert new
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
            return True
        except Exception as e:
            session.rollback()
            print(f"Error saving bond: {e}")
            return False
        finally:
            session.close()

    def save_bonds_batch(self, bonds: List[Bond], batch_size: int = 1000) -> int:
        """
        Batch save multiple bonds (optimized for bulk operations)

        Args:
            bonds: List of bonds to save
            batch_size: Number of bonds to commit per batch

        Returns:
            Number of bonds successfully saved
        """
        if not bonds:
            return 0

        session = self._get_session()
        saved_count = 0
        current_time = datetime.now().isoformat()

        try:
            # Get existing bond IDs in one query for efficiency
            existing_ids = set(
                session.query(BondModel.bond_id).filter(BondModel.bond_id.in_([b.bond_id for b in bonds])).all()
            )
            existing_ids = {id_tuple[0] for id_tuple in existing_ids}

            # Separate bonds into new and existing
            new_bonds = []
            update_map = {}

            for bond in bonds:
                if bond.bond_id in existing_ids:
                    update_map[bond.bond_id] = bond
                else:
                    new_bonds.append(bond)

            # Batch insert new bonds
            for i in range(0, len(new_bonds), batch_size):
                batch = new_bonds[i : i + batch_size]
                # Create dictionaries for bulk insert
                bond_dicts = [
                    {
                        "bond_id": bond.bond_id,
                        "bond_type": bond.bond_type.value,
                        "face_value": bond.face_value,
                        "coupon_rate": bond.coupon_rate,
                        "maturity_date": bond.maturity_date.isoformat(),
                        "issue_date": bond.issue_date.isoformat(),
                        "current_price": bond.current_price,
                        "credit_rating": bond.credit_rating,
                        "issuer": bond.issuer,
                        "frequency": bond.frequency,
                        "callable": bond.callable,
                        "convertible": bond.convertible,
                        "created_at": current_time,
                        "updated_at": current_time,
                    }
                    for bond in batch
                ]
                session.bulk_insert_mappings(BondModel, bond_dicts)
                saved_count += len(batch)

            # Batch update existing bonds
            if update_map:
                existing_bonds = session.query(BondModel).filter(BondModel.bond_id.in_(list(update_map.keys()))).all()

                for existing in existing_bonds:
                    bond = update_map[existing.bond_id]
                    existing.bond_type = bond.bond_type.value
                    existing.face_value = bond.face_value
                    existing.coupon_rate = bond.coupon_rate
                    existing.maturity_date = bond.maturity_date.isoformat()
                    existing.issue_date = bond.issue_date.isoformat()
                    existing.current_price = bond.current_price
                    existing.credit_rating = bond.credit_rating
                    existing.issuer = bond.issuer
                    existing.frequency = bond.frequency
                    existing.callable = bond.callable
                    existing.convertible = bond.convertible
                    existing.updated_at = current_time

                saved_count += len(update_map)

            session.commit()
            return saved_count

        except Exception as e:
            session.rollback()
            logger.error(f"Error in batch save: {e}")
            return saved_count
        finally:
            session.close()

    def load_bond(self, bond_id: str) -> Optional[Bond]:
        """Load bond from database"""
        session = self._get_session()
        try:
            bond_model = session.query(BondModel).filter(BondModel.bond_id == bond_id).first()
            if bond_model:
                return self._model_to_bond(bond_model)
            return None
        finally:
            session.close()

    def load_all_bonds(self) -> List[Bond]:
        """Load all bonds from database"""
        session = self._get_session()
        try:
            bond_models = session.query(BondModel).all()
            return [self._model_to_bond(bm) for bm in bond_models]
        finally:
            session.close()

    def _model_to_bond(self, model: BondModel) -> Bond:
        """Convert SQLAlchemy model to Bond object"""
        return Bond(
            bond_id=model.bond_id,
            bond_type=BondType(model.bond_type),
            face_value=model.face_value,
            coupon_rate=model.coupon_rate,
            maturity_date=datetime.fromisoformat(model.maturity_date),
            issue_date=datetime.fromisoformat(model.issue_date),
            current_price=model.current_price,
            credit_rating=model.credit_rating or "BBB",
            issuer=model.issuer or "",
            frequency=model.frequency or 2,
            callable=model.callable or False,
            convertible=model.convertible or False,
        )

    def save_price_history(self, bond_id: str, price: float, fair_value: Optional[float] = None):
        """Save price history entry"""
        session = self._get_session()
        try:
            price_history = PriceHistoryModel(
                bond_id=bond_id, price=price, fair_value=fair_value, timestamp=datetime.now().isoformat()
            )
            session.add(price_history)
            session.commit()
        except Exception as e:
            session.rollback()
            print(f"Error saving price history: {e}")
        finally:
            session.close()

    def get_price_history(self, bond_id: str, limit: int = 100) -> List[Dict]:
        """Get price history for a bond"""
        session = self._get_session()
        try:
            histories = (
                session.query(PriceHistoryModel)
                .filter(PriceHistoryModel.bond_id == bond_id)
                .order_by(PriceHistoryModel.timestamp.desc())
                .limit(limit)
                .all()
            )
            return [{"price": h.price, "fair_value": h.fair_value, "timestamp": h.timestamp} for h in histories]
        finally:
            session.close()

    def save_valuation(self, bond_id: str, fair_value: float, ytm: float, duration: float, convexity: float):
        """Save valuation data"""
        session = self._get_session()
        try:
            valuation = ValuationModel(
                bond_id=bond_id,
                fair_value=fair_value,
                ytm=ytm,
                duration=duration,
                convexity=convexity,
                timestamp=datetime.now().isoformat(),
            )
            session.add(valuation)
            session.commit()
        except Exception as e:
            session.rollback()
            print(f"Error saving valuation: {e}")
        finally:
            session.close()

    def save_arbitrage_opportunity(self, bond_id: str, profit_pct: float, recommendation: str):
        """Save arbitrage opportunity"""
        session = self._get_session()
        try:
            opportunity = ArbitrageOpportunityModel(
                bond_id=bond_id,
                profit_percentage=profit_pct,
                recommendation=recommendation,
                timestamp=datetime.now().isoformat(),
            )
            session.add(opportunity)
            session.commit()
        except Exception as e:
            session.rollback()
            print(f"Error saving arbitrage opportunity: {e}")
        finally:
            session.close()

    def delete_bond(self, bond_id: str) -> bool:
        """Delete bond from database"""
        session = self._get_session()
        try:
            bond_model = session.query(BondModel).filter(BondModel.bond_id == bond_id).first()
            if bond_model:
                session.delete(bond_model)
                session.commit()
                return True
            return False
        except Exception as e:
            session.rollback()
            print(f"Error deleting bond: {e}")
            return False
        finally:
            session.close()


# Backward compatibility: EnhancedBondDatabase is the default implementation
# The original BondDatabase class from data_persistence.py is deprecated
# Use EnhancedBondDatabase for all new code
BondDatabase = EnhancedBondDatabase  # Enhanced version is the default
