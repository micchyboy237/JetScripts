from jet.db.postgres import PostgresDB
from psycopg import OperationalError


def example_basic_usage():
    """Demonstrate basic database creation and deletion."""
    print("Example 1: Basic Database Creation and Deletion")
    db_utils = PostgresDB()
    db_name = "test_project_db"

    try:
        db_utils.create_db(db_name)
        print(f"Successfully created database: {db_name}")

        with db_utils.connect_db(db_name) as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT current_database();")
                result = cur.fetchone()
                print(f"Connected to database: {result['current_database']}")

    except Exception as e:
        print(f"Database operation failed: {e}")

    finally:
        try:
            db_utils.delete_db(db_name)
            print(f"Successfully deleted database: {db_name}")
        except Exception as e:
            print(f"Failed to delete database {db_name}: {e}")

    print("-" * 50)


def example_custom_config():
    """Demonstrate usage with custom configuration."""
    print("Example 2: Using Custom Configuration")
    db_utils = PostgresDB(
        default_db="postgres",
        user="jethroestrada",
        password="",
        host="localhost",
        port=5432
    )
    db_name = "custom_project_db"

    try:
        db_utils.create_db(db_name)
        print(f"Created database with custom config: {db_name}")

        with db_utils.connect_db(db_name) as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT version();")
                version = cur.fetchone()
                print(f"Connected to PostgreSQL: {version['version'][:20]}...")

    except Exception as e:
        print(f"Failed with custom config: {e}")

    finally:
        try:
            db_utils.delete_db(db_name)
            print(f"Deleted database: {db_name}")
        except Exception as e:
            print(f"Failed to delete database {db_name}: {e}")

    print("-" * 50)


def example_multiple_databases():
    """Demonstrate managing multiple databases."""
    print("Example 3: Managing Multiple Databases")
    db_utils = PostgresDB()
    db_names = ["app_db", "test_db", "staging_db"]

    created_dbs = []
    try:
        for db_name in db_names:
            db_utils.create_db(db_name)
            print(f"Created database: {db_name}")
            created_dbs.append(db_name)

        for db_name in created_dbs:
            with db_utils.connect_db(db_name) as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT current_database();")
                    result = cur.fetchone()
                    print(
                        f"Verified database exists: {result['current_database']}")

    except Exception as e:
        print(f"Multiple database operation failed: {e}")

    finally:
        for db_name in created_dbs:
            try:
                db_utils.delete_db(db_name)
                print(f"Deleted database: {db_name}")
            except Exception as e:
                print(f"Failed to delete database {db_name}: {e}")

    print("-" * 50)


if __name__ == "__main__":
    print("Running PostgresDB Usage Examples")
    print("=" * 50)

    try:
        example_basic_usage()
        example_custom_config()
        example_multiple_databases()
        print("All examples completed!")
    except Exception as e:
        print(f"Script execution failed: {e}")
