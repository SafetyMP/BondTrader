"""
Data Persistence Module
SQLite database for storing bond data, prices, and ML models
"""

import sqlite3
import json
from datetime import datetime
from typing import List, Dict, Optional
from pathlib import Path
from bondtrader.core.bond_models import Bond, BondType
import pickle


class BondDatabase:
    """SQLite database for bond data persistence"""
    
    def __init__(self, db_path: str = "bonds.db"):
        """Initialize database connection"""
        self.db_path = db_path
        self._init_database()
    
    def _init_database(self):
        """Initialize database tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Bonds table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS bonds (
                bond_id TEXT PRIMARY KEY,
                bond_type TEXT NOT NULL,
                face_value REAL NOT NULL,
                coupon_rate REAL NOT NULL,
                maturity_date TEXT NOT NULL,
                issue_date TEXT NOT NULL,
                current_price REAL NOT NULL,
                credit_rating TEXT,
                issuer TEXT,
                frequency INTEGER DEFAULT 2,
                callable INTEGER DEFAULT 0,
                convertible INTEGER DEFAULT 0,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Price history table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS price_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                bond_id TEXT NOT NULL,
                price REAL NOT NULL,
                fair_value REAL,
                timestamp TEXT NOT NULL,
                FOREIGN KEY (bond_id) REFERENCES bonds(bond_id)
            )
        """)
        
        # Valuations table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS valuations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                bond_id TEXT NOT NULL,
                fair_value REAL NOT NULL,
                ytm REAL,
                duration REAL,
                convexity REAL,
                timestamp TEXT NOT NULL,
                FOREIGN KEY (bond_id) REFERENCES bonds(bond_id)
            )
        """)
        
        # Arbitrage opportunities table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS arbitrage_opportunities (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                bond_id TEXT NOT NULL,
                profit_percentage REAL NOT NULL,
                recommendation TEXT,
                timestamp TEXT NOT NULL,
                FOREIGN KEY (bond_id) REFERENCES bonds(bond_id)
            )
        """)
        
        conn.commit()
        conn.close()
    
    def save_bond(self, bond: Bond) -> bool:
        """Save bond to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                INSERT OR REPLACE INTO bonds 
                (bond_id, bond_type, face_value, coupon_rate, maturity_date,
                 issue_date, current_price, credit_rating, issuer, frequency,
                 callable, convertible, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                bond.bond_id,
                bond.bond_type.value,
                bond.face_value,
                bond.coupon_rate,
                bond.maturity_date.isoformat(),
                bond.issue_date.isoformat(),
                bond.current_price,
                bond.credit_rating,
                bond.issuer,
                bond.frequency,
                1 if bond.callable else 0,
                1 if bond.convertible else 0,
                datetime.now().isoformat()
            ))
            conn.commit()
            return True
        except Exception as e:
            print(f"Error saving bond: {e}")
            return False
        finally:
            conn.close()
    
    def load_bond(self, bond_id: str) -> Optional[Bond]:
        """Load bond from database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT * FROM bonds WHERE bond_id = ?
        """, (bond_id,))
        
        row = cursor.fetchone()
        conn.close()
        
        if row:
            return self._row_to_bond(row)
        return None
    
    def load_all_bonds(self) -> List[Bond]:
        """Load all bonds from database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT * FROM bonds")
        rows = cursor.fetchall()
        conn.close()
        
        bonds = [self._row_to_bond(row) for row in rows]
        return bonds
    
    def _row_to_bond(self, row: tuple) -> Bond:
        """Convert database row to Bond object"""
        return Bond(
            bond_id=row[0],
            bond_type=BondType(row[1]),
            face_value=row[2],
            coupon_rate=row[3],
            maturity_date=datetime.fromisoformat(row[4]),
            issue_date=datetime.fromisoformat(row[5]),
            current_price=row[6],
            credit_rating=row[7] or "BBB",
            issuer=row[8] or "",
            frequency=row[9] or 2,
            callable=bool(row[10]),
            convertible=bool(row[11])
        )
    
    def save_price_history(self, bond_id: str, price: float, fair_value: Optional[float] = None):
        """Save price history entry"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO price_history (bond_id, price, fair_value, timestamp)
            VALUES (?, ?, ?, ?)
        """, (bond_id, price, fair_value, datetime.now().isoformat()))
        
        conn.commit()
        conn.close()
    
    def get_price_history(self, bond_id: str, limit: int = 100) -> List[Dict]:
        """Get price history for a bond"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT price, fair_value, timestamp
            FROM price_history
            WHERE bond_id = ?
            ORDER BY timestamp DESC
            LIMIT ?
        """, (bond_id, limit))
        
        rows = cursor.fetchall()
        conn.close()
        
        return [
            {'price': r[0], 'fair_value': r[1], 'timestamp': r[2]}
            for r in rows
        ]
    
    def save_valuation(self, bond_id: str, fair_value: float, ytm: float,
                      duration: float, convexity: float):
        """Save valuation data"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO valuations (bond_id, fair_value, ytm, duration, convexity, timestamp)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (bond_id, fair_value, ytm, duration, convexity, datetime.now().isoformat()))
        
        conn.commit()
        conn.close()
    
    def save_arbitrage_opportunity(self, bond_id: str, profit_pct: float,
                                   recommendation: str):
        """Save arbitrage opportunity"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO arbitrage_opportunities (bond_id, profit_percentage, recommendation, timestamp)
            VALUES (?, ?, ?, ?)
        """, (bond_id, profit_pct, recommendation, datetime.now().isoformat()))
        
        conn.commit()
        conn.close()
    
    def delete_bond(self, bond_id: str) -> bool:
        """Delete bond from database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute("DELETE FROM bonds WHERE bond_id = ?", (bond_id,))
            conn.commit()
            return True
        except Exception as e:
            print(f"Error deleting bond: {e}")
            return False
        finally:
            conn.close()
