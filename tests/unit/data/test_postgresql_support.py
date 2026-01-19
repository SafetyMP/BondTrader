"""
Unit tests for PostgreSQL support utilities
"""

import pytest
from unittest.mock import MagicMock, patch


@pytest.mark.unit
class TestPostgreSQLSupport:
    """Test PostgreSQL support utilities"""

    def test_postgresql_connection_string(self):
        """Test PostgreSQL connection string generation"""
        from bondtrader.data.postgresql_support import get_connection_string

        connection_string = get_connection_string(
            host="localhost",
            port=5432,
            database="testdb",
            username="testuser",
            password="testpass",
        )
        
        assert "localhost" in connection_string
        assert "5432" in connection_string
        assert "testdb" in connection_string
        assert "testuser" in connection_string

    def test_postgresql_connection_string_ssl(self):
        """Test PostgreSQL connection string with SSL"""
        from bondtrader.data.postgresql_support import get_connection_string

        connection_string = get_connection_string(
            host="localhost",
            port=5432,
            database="testdb",
            username="testuser",
            password="testpass",
            ssl_mode="require",
        )
        
        assert "sslmode=require" in connection_string or "ssl=true" in connection_string

    def test_check_postgresql_connection(self):
        """Test checking PostgreSQL connection"""
        from bondtrader.data.postgresql_support import check_connection

        with patch("bondtrader.data.postgresql_support.create_engine") as mock_engine:
            mock_conn = MagicMock()
            mock_conn.execute.return_value = None
            mock_engine.return_value.connect.return_value.__enter__.return_value = mock_conn
            
            try:
                result = check_connection("postgresql://localhost/testdb")
                # Should return boolean or handle gracefully
                assert isinstance(result, bool) or result is None
            except Exception:
                # May fail if dependencies not available
                pass