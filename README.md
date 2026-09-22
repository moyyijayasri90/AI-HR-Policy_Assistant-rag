# AI HR Policy Assistant – RAG

An AI-powered HR Policy Assistant that answers employee questions using Retrieval-Augmented Generation (RAG).

## Project Overview

This project allows employees to ask questions about company HR policies. The system retrieves relevant information from HR policy documents and uses Gemini to generate a clear answer with source references.

## Features

- HR policy question answering
- Retrieval-Augmented Generation (RAG)
- TF-IDF based document retrieval
- Cosine similarity for relevant document matching
- Gemini-powered answer generation
- Source document and page references
- Rejects unrelated questions
- Simple Streamlit web interface

## HR Policy Documents

The project uses the following HR policy documents:

- Employee Benefits
- Code of Conduct
- Leave Policy
- Remote Work Policy
- Working Hours Policy

## Technologies Used

- Python
- Streamlit
- Google Gemini API
- Scikit-learn
- TF-IDF
- Cosine Similarity
- PyPDF
- GitHub
- Kaggle

## How It Works

1. HR policy PDF documents are loaded.
2. Documents are divided into smaller text chunks.
3. TF-IDF converts the text into numerical vectors.
4. Cosine similarity finds the most relevant policy information.
5. Retrieved information is provided to Gemini.
6. Gemini generates an answer using the retrieved HR policy context.
7. The application displays the answer along with source references.

## Example

**Question:**  
How many annual leave days do employees receive?

**Answer:**  
Eligible employees receive 20 days of annual leave per calendar year.

**Source:** `leave_policy.pdf`

## Live Demo

[Open the AI HR Policy Assistant](https://ai-hr-policyassistant-rag-fw9bypuwephrp9vdjmrcww.streamlit.app/)

## Project Repository

[GitHub Repository](https://github.com/moyyijayasri90/AI-HR-Policy_Assistant-rag)

## Project Notebook

The RAG implementation and experiments were developed in Kaggle.

## Disclaimer

This project is a demonstration HR policy assistant using sample HR policy documents. It is not intended to provide legal or official HR advice.
