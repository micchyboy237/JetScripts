from jet.db.postgres import PostgresDB
from psycopg import OperationalError


def example_basic_usage():
    """Demonstrate basic database creation and deletion."""
    print("Example 1: Basic Database Creation and Deletion")
    # Initialize with default configuration
    db_utils = PostgresDB()

    db_name = "test_project_db"

    try:
        # Create a new database
        db_utils.create_db(db_name)
        print(f"Successfully created database: {db_name}")

        # Connect to the new database
        with db_utils.connect_db(db_name) as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT current_database();")
                result = cur.fetchone()
                print(f"Connected to database: {result[0]}")

        # Delete the database
        db_utils.delete_db(db_name)
        print(f"Successfully deleted database: {db_name}")

    except OperationalError as e:
        print(f"Database operation failed: {e}")

    print("-" * 50)


def example_custom_config():
    """Demonstrate usage with custom configuration."""
    print("Example 2: Using Custom Configuration")
    # Initialize with custom connection parameters
    db_utils = PostgresDB(
        default_db="postgres",
        user="custom_user",
        password="custom_password",
        host="localhost",
        port=5433
    )

    db_name = "custom_project_db"

    try:
        # Create database with custom config
        db_utils.create_db(db_name)
        print(f"Created database with custom config: {db_name}")

        # Verify connection
        with db_utils.connect_db(db_name) as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT version();")
                version = cur.fetchone()[0]
                print(f"Connected to PostgreSQL: {version[:20]}...")

        # Clean up
        db_utils.delete_db(db_name)
        print(f"Deleted database: {db_name}")

    except OperationalError as e:
        print(f"Failed with custom config: {e}")

    print("-" * 50)


def example_multiple_databases():
    """Demonstrate managing multiple databases."""
    print("Example 3: Managing Multiple Databases")
    db_utils = PostgresDB()

    db_names = ["app_db", "test_db", "staging_db"]

    try:
        # Create multiple databases
        for db_name in db_names:
            db_utils.create_db(db_name)
            print(f"Created database: {db_name}")

        # Connect to each and verify
        for db_name in db_names:
            with db_utils.connect_db(db_name) as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT current_database();")
                    result = cur.fetchone()
                    print(f"Verified database exists: {result[0]}")

        # Delete all databases
        for db_name in db_names:
            db_utils.delete_db(db_name)
            print(f"Deleted database: {db_name}")

    except OperationalError as e:
        print(f"Multiple database operation failed: {e}")

    print("-" * 50)


if __name__ == "__main__":
    print("Running PostgresDB Usage Examples")
    print("=" * 50)

    example_basic_usage()
    example_custom_config()
    example_multiple_databases()

    print("All examples completed!")
