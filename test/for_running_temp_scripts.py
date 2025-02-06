import re

# Define the query
query = 'MATCH (user)-[:HAS_PERSONAL_INFO]->(personal:Personal)'

# Use regex to find the node type inside parentheses
match = re.search(r'\((\w+)(?::\w+)?\)', query)

# If a match is found, extract the node type
if match:
    node_type = match.group(1)  # The first capturing group is the node type
    assert node_type == "user"

    # Update the query using the extracted node type
    updated_query = f"GRAPH_TEMPLATE.format(query=query, type={
        node_type.lower()})"
    print(updated_query)
else:
    print("No node type found in the query")
