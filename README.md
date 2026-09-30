# AI Chatbot - Task 2: RAG

A Retrieval-Augmented Generation (RAG) based chatbot developed using Python, Sentence Transformers, FAISS, and FLAN-T5.

## Objective

The objective of Task 2 is to improve the basic chatbot developed in Task 1 by introducing Retrieval-Augmented Generation (RAG).

Instead of depending only on predefined responses, the chatbot retrieves relevant information from a knowledge base and uses the retrieved information to generate an appropriate response.

## Technologies Used

- Python
- Sentence Transformers
- FAISS
- Hugging Face Transformers
- FLAN-T5
- PyTorch
- NumPy

## How RAG Works

The chatbot follows these steps:

1. The user enters a question.
2. The question is converted into a numerical embedding using Sentence Transformers.
3. FAISS searches the vector database for the most relevant information.
4. The retrieved information is combined into a context.
5. The context and user question are provided to the language model.
6. FLAN-T5 generates the final response.

## Architecture

```text
User Question
      ↓
Query Embedding
      ↓
Sentence Transformer
      ↓
FAISS Vector Search
      ↓
Relevant Documents
      ↓
Retrieved Context
      ↓
FLAN-T5
      ↓
Generated Response
