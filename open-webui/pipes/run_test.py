import asyncio
from visual_tree_of_thoughts import Node, Pipe, MCTS


async def main():
    # Create an initial node with a sample question
    root = Node(content="What are the benefits of regular exercise?")

    # Create the MCTS object
    pipe = Pipe()  # Mock or real LLM setup
    mcts = MCTS(
        question="What are the benefits of regular exercise?",
        root=root,
        llm=pipe,  # This should be connected to a mocked or real LLM
    )

    # Run the MCTS search
    best_node = await mcts.search(num_simulations=10)

    # Retrieve the best answer
    print("Best Answer Found:")
    print(best_node.content)

# Run the test script
asyncio.run(main())
