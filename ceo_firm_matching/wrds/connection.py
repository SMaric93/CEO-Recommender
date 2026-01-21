"""
WRDS Connection Utilities.

Provides centralized WRDS database connection management.
"""
import os
from typing import Optional

try:
    import wrds
    WRDS_AVAILABLE = True
except ImportError:
    WRDS_AVAILABLE = False


# Default credentials (can be overridden via environment variables)
DEFAULT_WRDS_USER = os.environ.get('WRDS_USER', 'maricste93')
DEFAULT_WRDS_PASS = os.environ.get('WRDS_PASS', 'jexvar-manryn-6Cosky')


def connect_wrds(username: Optional[str] = None, password: Optional[str] = None):
    """
    Establish WRDS connection.
    
    Args:
        username: WRDS username (defaults to env var or hardcoded)
        password: WRDS password (defaults to env var or hardcoded)
        
    Returns:
        wrds.Connection object
        
    Raises:
        ImportError: If wrds package is not installed
        ConnectionError: If connection fails
    """
    if not WRDS_AVAILABLE:
        raise ImportError(
            "wrds package not installed. Install with: pip install wrds"
        )
    
    user = username or DEFAULT_WRDS_USER
    pwd = password or DEFAULT_WRDS_PASS
    
    try:
        db = wrds.Connection(wrds_username=user, wrds_password=pwd)
        print(f"✓ Connected to WRDS as {user}")
        return db
    except Exception as e:
        raise ConnectionError(f"Failed to connect to WRDS: {e}")


def test_wrds_access(db, schemas: Optional[list] = None) -> dict:
    """
    Test access to required WRDS schemas.
    
    Args:
        db: WRDS connection object
        schemas: List of schemas to test (defaults to common schemas)
        
    Returns:
        Dict mapping schema name to (accessible: bool, table_count: int)
    """
    if schemas is None:
        schemas = [
            'boardex',
            'ciq_pplintel',
            'execcomp',
            'crsp',
            'comp',
        ]
    
    results = {}
    for schema in schemas:
        try:
            tables = db.list_tables(library=schema)
            results[schema] = (True, len(tables))
            print(f"  ✓ {schema}: {len(tables)} tables")
        except Exception as e:
            results[schema] = (False, 0)
            print(f"  ✗ {schema}: {e}")
    
    return results
