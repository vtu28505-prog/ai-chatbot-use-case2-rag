# =========================================================
# TASK 2 - RAG CHATBOT
# =========================================================

import os
import numpy as np
import faiss

from sentence_transformers import SentenceTransformer
from transformers import pipeline


# =========================================================
# 1. KNOWLEDGE BASE
# =========================================================

knowledge_base = [
    "Artificial Intelligence (AI) is a field of computer science that enables machines to perform tasks that normally require human intelligence.",
    
    "Machine Learning (ML) is a subset of artificial intelligence that allows computers to learn patterns from data and make predictions or decisions.",
    
    "Deep Learning is a branch of machine learning that uses neural networks with multiple layers to learn complex patterns from data.",
    
    "Cybersecurity is the practice of protecting computers, networks, applications, and data from unauthorized access and cyber attacks.",
    
    "Phishing is a cyber attack in which attackers use fraudulent emails, messages, or websites to trick users into revealing sensitive information.",
    
    "Malware is malicious software designed to damage systems, steal information, disrupt operations, or gain unauthorized access.",
    
    "Ransomware is a type of malware that encrypts files or restricts access to a system and demands payment from the victim.",
    
    "Encryption is the process of converting readable information into an encoded form so that unauthorized users cannot understand it.",
    
    "A firewall is a security mechanism that monitors and controls incoming and outgoing network traffic based on predefined security rules.",
    
    "A chatbot is a software application that communicates with users using natural language and can provide information or perform specific tasks."
]


# =========================================================
# 2. LOAD EMBEDDING MODEL
# =========================================================

print("Loading embedding model...")

embedding_model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)


# =========================================================
# 3. CREATE EMBEDDINGS
# =========================================================

print("Creating knowledge base embeddings...")

embeddings = embedding_model.encode(
    knowledge_base,
    convert_to_numpy=True
)

print("Embedding shape:", embeddings.shape)


# =========================================================
# 4. CREATE FAISS VECTOR DATABASE
# =========================================================

dimension = embeddings.shape[1]

index = faiss.IndexFlatL2(dimension)

index.add(
    embeddings.astype("float32")
)

print("FAISS index created.")
print("Number of stored documents:", index.ntotal)


# =========================================================
# 5. LOAD GENERATIVE MODEL
# =========================================================

print("Loading language model...")

generator = pipeline(
    "text2text-generation",
    model="google/flan-t5-base",
    max_new_tokens=150
)


# =========================================================
# 6. RETRIEVAL FUNCTION
# =========================================================

def retrieve_context(
    query,
    top_k=3
):

    # Convert query into embedding
    query_embedding = embedding_model.encode(
        [query],
        convert_to_numpy=True
    )

    # Search FAISS
    distances, indices = index.search(
        query_embedding.astype("float32"),
        top_k
    )

    retrieved_documents = []

    for i in indices[0]:

        if i < len(knowledge_base):

            retrieved_documents.append(
                knowledge_base[i]
            )

    return retrieved_documents


# =========================================================
# 7. RAG CHATBOT
# =========================================================

def chatbot(user_query):

    # Retrieve relevant information
    retrieved_documents = retrieve_context(
        user_query,
        top_k=3
    )

    # Combine retrieved information
    context = "\n".join(
        retrieved_documents
    )

    # Create prompt
    prompt = f"""
Answer the user's question using the provided context.

Context:
{context}

Question:
{user_query}

Answer:
"""

    # Generate response
    result = generator(prompt)

    response = result[0]["generated_text"]

    return response


# =========================================================
# 8. CHAT INTERFACE
# =========================================================

print("=" * 60)
print("RAG CHATBOT")
print("=" * 60)

print("Ask questions about the knowledge base.")
print("Type 'bye' to exit.")
print()


while True:

    user_input = input("You: ")

    if user_input.lower().strip() in [
        "bye",
        "goodbye",
        "exit",
        "quit"
    ]:

        print("Bot: Goodbye!")
        break

    if not user_input.strip():

        print("Bot: Please enter a question.")
        continue

    response = chatbot(
        user_input
    )

    print("Bot:", response)
    print()
