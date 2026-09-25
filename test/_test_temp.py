"""
E2E tests for PostgreSQL ENUM type handling.

Tests creating enum types, using them in tables, and verifying data integrity.
Uses automatic database creation and cleanup for isolation.
"""

import uuid

import psycopg
import pytest
from jet.db.postgres.config import (
    DEFAULT_HOST,
    DEFAULT_PASSWORD,
    DEFAULT_PORT,
    DEFAULT_USER,
)
from psycopg import sql

# Test configuration - adjust these to match your local setup
BASE_DB_CONFIG = {
    "dbname": "test_enum_db",
    "user": DEFAULT_USER,
    "password": DEFAULT_PASSWORD,
    "host": DEFAULT_HOST,
    "port": DEFAULT_PORT,
}


def generate_test_db_name():
    """Generate a unique database name for test isolation."""
    return f"test_enum_{uuid.uuid4().hex[:8]}"


@pytest.fixture(scope="session")
def test_database():
    """Create a unique test database for the entire test session."""
    db_name = generate_test_db_name()

    # Create the test database
    admin_conn = psycopg.connect(**BASE_DB_CONFIG)
    admin_conn.autocommit = True

    try:
        with admin_conn.cursor() as cur:
            cur.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(db_name)))
        print(f"\n✓ Created test database: {db_name}")
    except Exception as e:
        print(f"\n✗ Failed to create test database: {e}")
        raise
    finally:
        admin_conn.close()

    # Return database info for tests to use
    test_db_config = BASE_DB_CONFIG.copy()
    test_db_config["dbname"] = db_name

    yield test_db_config

    # Cleanup: Drop the test database
    try:
        admin_conn = psycopg.connect(**BASE_DB_CONFIG)
        admin_conn.autocommit = True

        # Terminate existing connections to the test database
        with admin_conn.cursor() as cur:
            cur.execute(
                """
                SELECT pg_terminate_backend(pid) 
                FROM pg_stat_activity 
                WHERE datname = %s AND pid <> pg_backend_pid()
            """,
                (db_name,),
            )

        # Drop the database
        with admin_conn.cursor() as cur:
            cur.execute(
                sql.SQL("DROP DATABASE IF EXISTS {}").format(sql.Identifier(db_name))
            )
        print(f"✓ Cleaned up test database: {db_name}")

        admin_conn.close()
    except Exception as e:
        print(f"Warning: Failed to cleanup test database {db_name}: {e}")


@pytest.fixture
def db_connection(test_database):
    """Create a database connection for each test."""
    conn = psycopg.connect(**test_database)
    conn.autocommit = True
    yield conn
    conn.close()


@pytest.fixture(autouse=True)
def cleanup_enums_and_tables(db_connection):
    """Automatically clean up enum types and tables after each test."""
    # Track created types and tables for cleanup
    created_types = []
    created_tables = []

    # Store references in the connection for tracking
    db_connection._test_created_types = created_types
    db_connection._test_created_tables = created_tables

    yield

    # Cleanup: Drop tables first (they depend on enum types)
    for table_name in reversed(created_tables):
        try:
            with db_connection.cursor() as cur:
                cur.execute(
                    sql.SQL("DROP TABLE IF EXISTS {} CASCADE").format(
                        sql.Identifier(table_name)
                    )
                )
        except Exception as e:
            print(f"Warning: Failed to drop table {table_name}: {e}")

    # Cleanup: Drop enum types
    for type_name in reversed(created_types):
        try:
            with db_connection.cursor() as cur:
                cur.execute(
                    sql.SQL("DROP TYPE IF EXISTS {} CASCADE").format(
                        sql.Identifier(type_name)
                    )
                )
        except Exception as e:
            print(f"Warning: Failed to drop type {type_name}: {e}")


def create_enum_type(conn, type_name, values):
    """Helper to create an enum type and track it for cleanup."""
    with conn.cursor() as cur:
        # Format enum values properly
        values_sql = ", ".join([f"'{v}'" for v in values])
        query = sql.SQL("CREATE TYPE {} AS ENUM ({})").format(
            sql.Identifier(type_name), sql.SQL(values_sql)
        )
        cur.execute(query)

    # Track for cleanup
    conn._test_created_types.append(type_name)


def create_table_with_enum(conn, table_name, columns):
    """Helper to create a table with enum columns and track it for cleanup."""
    with conn.cursor() as cur:
        # Build column definitions
        col_defs = []
        for col_name, col_type in columns.items():
            col_defs.append(
                sql.SQL("{} {}").format(sql.Identifier(col_name), sql.SQL(col_type))
            )

        columns_sql = sql.SQL(", ").join(col_defs)
        query = sql.SQL("CREATE TABLE {} ({})").format(
            sql.Identifier(table_name), columns_sql
        )
        cur.execute(query)

    # Track for cleanup
    conn._test_created_tables.append(table_name)


class TestEnumBasicOperations:
    """Test basic enum type creation and usage."""

    def test_create_enum_type(self, db_connection):
        """Test creating a simple enum type."""
        create_enum_type(db_connection, "test_mood", ["sad", "ok", "happy"])

        # Verify the type exists
        with db_connection.cursor() as cur:
            cur.execute("""
                SELECT typname FROM pg_type 
                WHERE typname = 'test_mood' AND typtype = 'e'
            """)
            result = cur.fetchone()
            assert result is not None
            assert result[0] == "test_mood"

    def test_create_table_with_enum_column(self, db_connection):
        """Test creating a table that uses an enum type."""
        create_enum_type(db_connection, "priority_level", ["low", "medium", "high"])
        create_table_with_enum(
            db_connection,
            "tasks",
            {
                "id": "SERIAL PRIMARY KEY",
                "title": "TEXT NOT NULL",
                "priority": "priority_level",
            },
        )

        # Verify table structure
        with db_connection.cursor() as cur:
            cur.execute("""
                SELECT column_name, data_type, udt_name 
                FROM information_schema.columns 
                WHERE table_name = 'tasks'
                ORDER BY ordinal_position
            """)
            columns = cur.fetchall()

            # Check priority column uses our enum
            priority_col = [c for c in columns if c[0] == "priority"][0]
            assert priority_col[1] == "USER-DEFINED"
            assert priority_col[2] == "priority_level"

    def test_insert_enum_values(self, db_connection):
        """Test inserting data with enum values."""
        create_enum_type(db_connection, "status", ["pending", "active", "completed"])
        create_table_with_enum(
            db_connection,
            "projects",
            {"id": "SERIAL PRIMARY KEY", "name": "TEXT", "status": "status"},
        )

        # Insert rows with different enum values
        with db_connection.cursor() as cur:
            cur.execute("""
                INSERT INTO projects (name, status) VALUES 
                ('Project A', 'pending'),
                ('Project B', 'active'),
                ('Project C', 'completed')
            """)

        # Verify inserts
        with db_connection.cursor() as cur:
            cur.execute("SELECT name, status FROM projects ORDER BY id")
            rows = cur.fetchall()

            assert len(rows) == 3
            assert rows[0] == ("Project A", "pending")
            assert rows[1] == ("Project B", "active")
            assert rows[2] == ("Project C", "completed")

    def test_query_enum_values(self, db_connection):
        """Test querying data using enum values in WHERE clause."""
        create_enum_type(db_connection, "color", ["red", "green", "blue"])
        create_table_with_enum(
            db_connection,
            "items",
            {"id": "SERIAL PRIMARY KEY", "name": "TEXT", "color": "color"},
        )

        # Insert test data
        with db_connection.cursor() as cur:
            cur.execute("""
                INSERT INTO items (name, color) VALUES 
                ('Item 1', 'red'),
                ('Item 2', 'green'),
                ('Item 3', 'blue'),
                ('Item 4', 'red')
            """)

        # Query by enum value
        with db_connection.cursor() as cur:
            cur.execute("SELECT name FROM items WHERE color = 'red' ORDER BY id")
            red_items = cur.fetchall()

            assert len(red_items) == 2
            assert red_items[0][0] == "Item 1"
            assert red_items[1][0] == "Item 4"


class TestEnumConstraints:
    """Test enum type constraints and validation."""

    def test_invalid_enum_value_rejected(self, db_connection):
        """Test that invalid enum values are rejected."""
        create_enum_type(db_connection, "size", ["small", "medium", "large"])
        create_table_with_enum(
            db_connection,
            "products",
            {"id": "SERIAL PRIMARY KEY", "name": "TEXT", "size": "size"},
        )

        # Try to insert invalid enum value
        with db_connection.cursor() as cur:
            with pytest.raises(psycopg.errors.InvalidTextRepresentation):
                cur.execute("""
                    INSERT INTO products (name, size) VALUES 
                    ('Product X', 'extra-large')
                """)

    def test_case_sensitive_enum_values(self, db_connection):
        """Test that enum values are case-sensitive."""
        create_enum_type(db_connection, "level", ["Basic", "Premium", "Enterprise"])
        create_table_with_enum(
            db_connection,
            "subscriptions",
            {"id": "SERIAL PRIMARY KEY", "user": "TEXT", "level": "level"},
        )

        # Insert with correct case
        with db_connection.cursor() as cur:
            cur.execute("""
                INSERT INTO subscriptions (user, level) VALUES 
                ('User1', 'Basic')
            """)

        # Try to insert with wrong case - should fail
        with db_connection.cursor() as cur:
            with pytest.raises(psycopg.errors.InvalidTextRepresentation):
                cur.execute("""
                    INSERT INTO subscriptions (user, level) VALUES 
                    ('User2', 'basic')
                """)


class TestEnumOrdering:
    """Test enum value ordering behavior."""

    def test_enum_order_follows_declaration(self, db_connection):
        """Test that enum ordering follows declaration order."""
        create_enum_type(db_connection, "rating", ["poor", "fair", "good", "excellent"])
        create_table_with_enum(
            db_connection,
            "reviews",
            {"id": "SERIAL PRIMARY KEY", "product": "TEXT", "rating": "rating"},
        )

        # Insert reviews in random order
        with db_connection.cursor() as cur:
            cur.execute("""
                INSERT INTO reviews (product, rating) VALUES 
                ('Product A', 'good'),
                ('Product B', 'poor'),
                ('Product C', 'excellent'),
                ('Product D', 'fair')
            """)

        # Order by enum should follow declaration order
        with db_connection.cursor() as cur:
            cur.execute("SELECT product, rating FROM reviews ORDER BY rating")
            ordered_reviews = cur.fetchall()

            expected_order = [
                ("Product B", "poor"),
                ("Product D", "fair"),
                ("Product A", "good"),
                ("Product C", "excellent"),
            ]

            assert ordered_reviews == expected_order


class TestMultipleEnumTypes:
    """Test using multiple enum types in one table."""

    def test_table_with_multiple_enums(self, db_connection):
        """Test a table with multiple enum columns."""
        create_enum_type(
            db_connection, "department", ["engineering", "sales", "marketing"]
        )
        create_enum_type(
            db_connection, "employment_status", ["full-time", "part-time", "contract"]
        )

        create_table_with_enum(
            db_connection,
            "employees",
            {
                "id": "SERIAL PRIMARY KEY",
                "name": "TEXT",
                "department": "department",
                "status": "employment_status",
            },
        )

        # Insert employee data
        with db_connection.cursor() as cur:
            cur.execute("""
                INSERT INTO employees (name, department, status) VALUES 
                ('Alice', 'engineering', 'full-time'),
                ('Bob', 'sales', 'part-time'),
                ('Charlie', 'marketing', 'contract')
            """)

        # Query and verify
        with db_connection.cursor() as cur:
            cur.execute("SELECT name, department, status FROM employees ORDER BY id")
            employees = cur.fetchall()

            assert employees[0] == ("Alice", "engineering", "full-time")
            assert employees[1] == ("Bob", "sales", "part-time")
            assert employees[2] == ("Charlie", "marketing", "contract")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "-s"])
