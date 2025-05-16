# Simple RAG with RL

[![Python 3.7+](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/release/python-370/) [![Nebius AI](https://img.shields.io/badge/Nebius%20AI-LLM-brightgreen)](https://cloud.nebius.ai/services/llm-embedding) [![OpenAI](https://img.shields.io/badge/OpenAI-API-lightgrey)](https://openai.com/) [![Medium](https://img.shields.io/badge/Medium-Blog-black?logo=medium)](https://medium.com/@fareedkhandev/maximizing-simple-rag-performance-using-rl-rewards-in-python-d4c14cbadf59)

A simple RAG works in three simple steps:

1. **Indexing**: Break documents into chunks and convert to vector embeddings.

2. **Retrieval**: When a question is asked, find the most relevant chunks.

3. **Generation**: Combine the question with retrieved chunks and let the AI generate an answer using this information.

The actual problem is to generate an answer to a given question using the provided documents. Simple RAG often fails to generate accurate answers due to the lack of context in the retrieved chunks. In this notebook, we will use the `RL RAG` approach to generate answers to the given questions using the provided documents.

# Table of Contents

- [Setting Up the Environment](#setting-up-the-environment)
- [Data Preprocessing](#data-preprocessing)
- [Document Embedding Generation](#document-embedding-generation)
- [Vector Store Implementation](#vector-store-implementation)
- [Simple Retrieval Implementation](#simple-retrieval-implementation)
  - [Cosine Similarity](#cosine-similarity)
  - [Similarity Search](#similarity-search)
  - [LLM Response Generation](#llm-response-generation)
  - [Basic RAG Pipeline](#basic-rag-pipeline)
  - [Evaluation of Basic RAG](#evaluate-the-basic-rag-pipeline)
- [Reinforcement Learning for RAG](#reinforcement-learning-for-rag)
  - [State, Action Space, and Reward Methodology](#state-action-space-and-reward-methodology)
  - [Policy Network](#policy-network)
  - [Single RL Step](#single-rl-step)
  - [Training Parameters and Policy Update](#training-parameters-and-policy-update)
  - [Training Loop](#training-loop)
  - [Performance Comparison Logic](#performance-comparison-logic)
- [Evaluation Framework](#evaluation-framework)
- [Evaluating RL vs Simple RAG](#evaluating-rl-vs-simple-rag)
- [Saving Comparison Results](#saving-the-comparison-results)
- [Conclusion](#what-can-we-conclude)

## Setting Up the Environment

First, we need to import the necessary libraries and set up the environment. We will be using HuggingFace Models hosted under **Nebius** platform. Obviously, you can use your own models as long as they are compatible with OpenAI's API.

Next, we need to initialize the client responsible for response and embedding generation.

## Data Preprocessing
Now that we have moved onto the data preprocessing stage, we need to load the data and preprocess it. Let's create a function that will load all the `.txt` files from a directory and return a list of documents.

We need to create a function that performs chunking of the documents once they are loaded. We are using a `chunk_size` of `100` characters, but you can adjust it as per your requirements.

This step is **optional**, where we preprocess each chunk by removing special characters, converting to lowercase, etc.

However, if you are using the previous preprocessing step, you can simply create a function to preprocess the entire document.

Now that we have implemented all the functions for data preprocessing, we can load the documents from the directory, split them into chunks, and preprocess the chunks.

Print the first 200 characters of the first two chunks

## Document Embedding Generation

In the previous step, we chunked our document. Now it's time to generate embeddings for the chunk dataset. When working with RAG, our knowledge base is typically quite large. Therefore, we need to perform embedding generation in batches. Let's create a core function to generate embeddings for the chunks in batches.

The embedding model we are using is `BAAI/bge-en-icl`.

Next, we will define a function to generate embeddings for all text chunks in batches. This function will take a list of text chunks as input and generate embeddings for each batch of chunks using the OpenAI client. The function will return a list of embeddings corresponding to all the text chunks.

Let's create another function to save the embeddings to a file in JSON format.

Now that we have implemented all the functions for embedding generation, we can proceed to generate embeddings for the preprocessed text chunks and save them to a JSON file.

## Vector Store Implementation
Since we are not using any python libraries for vector storage, we will implement a simple vector store using a dictionary.

## Simple Retrieval Implementation

We do know for retrieving the most similar text chunks to a given query, we can use the cosine similarity between the query embedding and the embeddings of all text chunks. The higher the cosine similarity, the more similar the text chunks are. We can then sort the chunks based on their similarity scores and return the top-k most similar chunks.

So, let's implement a simple cosine similarity-based retrieval function.

The cosine similarity between two vectors $A$ and $B$ is calculated as:

$$\text{cosine similarity} = \frac{A \cdot B}{||A|| \times ||B||} = \frac{\sum_{i=1}^{n} A_i B_i}{\sqrt{\sum_{i=1}^{n} A_i^2} \times \sqrt{\sum_{i=1}^{n} B_i^2}}$$

Where:
- $A \cdot B$ is the dot product of vectors $A$ and $B$
- $||A||$ and $||B||$ are the Euclidean norms (magnitudes) of the vectors
- $n$ is the dimension of the vectors

When we calculate the cosine similarity between a query and all the chunks, we can perform a similarity search. Based on the `top_k` parameter, we retrieve the top k most similar chunks.

Once we have the similarity search function ready, we can simply code a retrieval function on top of it that will provide the relevant chunks based on the query.

Now that we have implemented all the functions for retrieval, we can proceed to test the retrieval system with a sample query.

## LLM Response Generation

When we have a query and a set of relevant document chunks, we can use a large language model (LLM) to generate a response based on the query and the retrieved information. In this section, we will use the OpenAI API to generate a response to a query by providing the query text and the relevant document chunks as context to the LLM.

First we need a function to construct the input prompt for the LLM, which includes the query text and the relevant document chunks as context.

To generate an LLM response, we need to implement a function that takes the constructed input prompt and sends it to the OpenAI API for response generation.

## Basic RAG Pipeline

We cannot run small pieces of code repeatedly. Therefore, we need to create a simple RAG pipeline that takes only one parameter, which is our query, and returns the LLM response.

## Evaluate the basic RAG pipeline

Now that we have coded the basic RAG pipeline, we can use it for evaluation. Our evaluation queries contain different targeted segments, such as `factual_queries` and `complex_nature`. We are going to test the factual knowledge of our RAG pipeline.

Let's load our evaluation queries and their expected answers.

Let's test the basic RAG pipeline with this eval query and see how well it performs.

The simple RAG pipeline doesn't seem to perform well in its current state. The generated response is not only irrelevant to the ground truth but also misses critical information.

But don't worry! In the upcoming steps, we will implement a Reinforcement Learning-based RAG pipeline to address these shortcomings. This will help us improve the retrieval and generation process, making the responses more accurate and contextually relevant.

Stay tuned as we take our RAG pipeline to the next level! 🚀

## Reinforcement Learning for RAG

Reinforcement Learning (RL) is a type of machine learning where an agent learns to make decisions by taking actions in an environment to maximize some notion of cumulative reward. Unlike supervised learning, the agent is not explicitly told which actions to take, but instead must discover which actions yield the most reward through trial and error.

Follow are the main components of a reinforcement learning system:

1. **Agent**: The learner or decision-maker
2. **Environment**: The world with which the agent interacts
3. **State (S)**: The current situation of the agent in the environment
4. **Action (A)**: A set of possible moves the agent can make
5. **Reward (R)**: Feedback from the environment after each action
6. **Policy (π)**: Strategy that the agent follows to determine the next action

The goal in reinforcement learning is to learn a policy π that maximizes the expected cumulative reward:

$$\pi^* = \arg\max_\pi \mathbb{E}\left[ \sum_{t=0}^{T} \gamma^t R_t \right]$$

Where:
- $\pi^*$ is the optimal policy
- $\gamma$ is the discount factor (0 ≤ γ ≤ 1)
- $R_t$ is the reward at time step t
- $T$ is the final time step

In the context of RAG systems, reinforcement learning can be used to:
- Improve retrieval by learning which documents are most helpful
- Refine prompt construction based on user feedback
- Optimize the generation process by learning from successful responses

## State, Action Space, and Reward Methodology

The very first step when coding an RL algorithm is to define three things:

- **State**: It is the current situation of the environment. In our case, the initial state is our simple RAG pipeline (query, context, response).
- **Action Space**: It is the decision that the agent takes based on the state. In our case, the actions can include changing the model, modifying the context, altering the query, etc.
- **Reward**: It is the feedback that the agent receives after taking an action. In our case, the reward can be the similarity between the generated response and the ground truth answer.

Our state will be changing constantly as we perform training. For that, we need to save the state after each `training episode` so that our RL agent can learn from it and avoid making the same mistakes again.

We have defined the state representation for the RL agent, including the user query, retrieved context chunks, rewritten query (if any), and histories of responses and rewards. This state will guide the agent in generating better responses.

Next we need to define the action space for the reinforcement learning agent. The action space consists of the set of possible actions that the agent can take at each step. In this case, we define four actions:
- `rewrite_query`: Reformulate the original query to improve retrieval
- `expand_context`: Retrieve additional context chunks
- `filter_context`: Remove irrelevant context chunks
- `generate_response`: Generate a response based on the current query and context

Obviously, when our RL agent takes an action, it will be based on the current state and the action space. It will be rewarded based on the quality of the response generated by the RAG pipeline. The reward function will be based on the cosine similarity between the generated response and the ground truth answer.

Our goal is to maximize the reward by generating responses that are similar to the ground truth answer. Higher reward values indicate that the generated response is more aligned with the expected answer.

## Action Function Logic

Now that we have defined the action space, we need to implement the logic for each action. This logic will determine how the RAG pipeline should be modified based on the action taken by the RL agent.

Just to revisit, the four actions are:
- `rewrite_query`: Reformulate the original query to improve retrieval
- `expand_context`: Retrieve additional context chunks
- `filter_context`: Remove irrelevant context chunks
- `generate_response`: Generate a response based on the current query and context

Let's create our first action logic for the agent. The first action we will implement is the `rewrite_query` action, which involves reformulating the original user query to improve retrieval performance. This action is crucial for enhancing the relevance of the retrieved context and generating more accurate responses.

This action is crucial for enhancing the relevance of the retrieved context and generating more accurate responses.

Let's code our next action logic, which is to expand the context by retrieving additional chunks. We will use the existing function `retrieve_relevant_chunks` to get more context chunks and then filter out any duplicates from the current context. We will limit the number of new chunks to be added to the context to a specified top_k value.

We need to filter the context to keep only the most relevant chunks for the query. This filtering step is crucial to ensure that the context provided to the language model is concise and focused on the most relevant information.

This action will help the agent explore more information relevant to the query.

## Policy Network

Previously, we defined our state, actions, and reward logic. Next, we need to create a policy network that will select an action based on the current state.

A policy network is a function that takes the current state and the action space as input and returns the selected action based on the state.

The policy network can use a simple heuristic to select an action based on the current state. For example, if there are no previous responses, the policy network can prioritize rewriting the query. If the context has too many chunks, the policy network can choose to filter the context.

So our policy network works like this:
- If there are no previous responses, prioritize rewriting the query.
- If there are previous responses but the rewards are low, try expanding the context.
- If the context has too many chunks, try filtering the context.
- Otherwise, generate a response.

## Single RL Step

We have coded an important component of the RL pipeline. For any developer who has done any kind of training, there exists a training loop where each iteration is a single step in which the RL agent takes an action, rewards are calculated, states are updated, and so on. So, we need to code a single step of our training loop. Let's do that.

In our single step function, we first select an action using the policy network. The policy network uses an epsilon-greedy strategy to balance exploration and exploitation. If the random number is less than epsilon, we choose a random action from the action space for exploration. Otherwise, we select the best action based on the current state using a simple heuristic.

## Training Parameters and Policy Update

We need to define some training parameters for our training loop and also define a function to update the policy based on the rewards received.

Though the training parameters function is **optional**, it can be used for advanced implementations of the RL pipeline.

Similar to how our state changes after each step in the RL process, the policy also needs to be updated based on the rewards received. The update_policy function takes the current policy, state, action, reward, and learning rate as input and returns the updated policy.

In the above `update_policy` logic, we store the action taken and the reward received for each query in the policy dictionary. In a more advanced RL algorithm, the policy update would involve more sophisticated methods such as policy gradients or Q-learning.

Finally, we need to implement progress tracking logic to monitor the training process. This will help us understand how the model is learning and improving over time.

## Training Loop

Now that we have coded every part of the training loop, we can put it all together in a single function that implements the training loop for the RL-enhanced RAG system.

This function will take the input query text, the expected ground truth answer, and optionally some training parameters. It will return the updated policy, a list of rewards received in each episode, a list of actions taken in each episode, and the best response generated during training.

In more detail, the `training_loop` function will:
- Initialize training parameters if not provided.
- Get the initial performance from the simple RAG pipeline for comparison.
- Start the training loop for the specified number of episodes.
- Perform a single RL step in each episode.
- Update rewards and actions history for each episode.
- Print progress every 5 episodes.
- Compare the best RL-enhanced RAG reward with the simple RAG reward.
- Return the updated policy, rewards history, actions history, and the best response generated during training.

## Performance Comparison Logic

Although we can manually compare the simple RAG pipeline with the RL-based RAG pipeline, a function can definitely help us in this regard. So, let's define a function to compare the performance of the simple RAG pipeline with the RL-enhanced RAG pipeline.

So our performance comparison logic is not very complicated but is based on 4 steps:
1. Generate a response using the simple RAG pipeline.
2. Train the RL-enhanced RAG model using the training loop.
3. Evaluate and compare the results.
4. Plot the reward history (if available).

## Evaluation Framework (**Optional**)

This step is optional but in case you want to evaluate all the eval queries in the validation data, you can use the following code.

First, to check the relevance of the retrieved chunks and the ground truth, we need to have a function that evaluates the relevance of the retrieved chunks.

To evaluate the accuracy of the generated responses, we can use the cosine similarity between the embeddings of the generated responses and the ground truth. So let's define a function to evaluate the accuracy of the responses based on this similarity metric.

We also need to measure the response quality and assign a relevant score for it to be used in the reinforcement learning process.

Then we can evaluate the performance of the RL-enhanced RAG model on the validation dataset:

## Evaluating (RL vs Simple) RAG

Ah, the moment of truth! Let's evaluate the performance of the simple RAG pipeline against the RL-enhanced RAG pipeline on our factual query, where the simple RAG previously failed to provide the correct answer. Let's see if the RL-enhanced RAG pipeline can perform better.

Let's revisit our evaluation query and see what the simple RAG pipeline generates for it.

You can clearly see that the response generated by the RL-enhanced RAG model is more accurate and relevant compared to the simple RAG pipeline. The improvement in similarity to the ground truth is evident, indicating that the RL-enhanced model has learned to generate better responses through training.

## Saving the Comparison Results

After implementing the RL algorithm, we can save the comparison results to check the performance of the RL implementation later.

## What can we conclude?

- The performance of the simple RAG is lower compared to the RL-enhanced RAG on factual queries.
- The RL-enhanced RAG achieved a 19.5% improvement in the similarity score within 5 episodes.
- Further improvements can be achieved by:
    - Training for more episodes.
    - Tuning hyperparameters.
- Time is a key constraint for training.
- Parallel implementation of the RL algorithm can help reduce training time.
