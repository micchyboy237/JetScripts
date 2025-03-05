from jet.db.pgvector import PgVectorClient

# Initialize the client
client = PgVectorClient(
    dbname='vector_db1',
    user='jethroestrada',
    password='',
    host='localhost',
    port=5432
)


# Create a table for storing 3D vectors
client.create_table("items", 3)

# Insert a vector
vector_id = client.insert_vector("items", [1.0, 2.0, 3.0])
print(f"Inserted vector with ID: {vector_id}")

# Search for similar vectors
similar_vectors = client.search_similar("items", [1.1, 2.1, 3.1], top_k=3)
print("Similar vectors:", similar_vectors)

# Drop all rows in "items" table
client.drop_all_rows("items")

# Delete all tables in the database
client.delete_all_tables()

# Close connection
client.close()
