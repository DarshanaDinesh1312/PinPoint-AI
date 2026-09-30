# 📌 PinPoint AI

## Intelligent PDF Understanding & Semantic Search System

## 📌 Project Overview

PinPoint AI is an AI-powered document understanding system that converts PDF files and books into searchable knowledge.

The project extracts document text, cleans and chunks the content, generates embeddings, stores them in a vector index, and retrieves relevant sections using semantic search.

It is designed as the foundation for a complete Retrieval-Augmented Generation (RAG) application.

## 🎯 Objectives

- Upload and process PDF documents
- Extract text using PyMuPDF
- Clean and preprocess document content
- Split long documents into meaningful chunks
- Generate text embeddings
- Store embeddings using FAISS
- Perform semantic document search
- Prepare retrieved content for RAG-based question answering
- Build an interactive Streamlit interface

## 🔄 Project Workflow

```text
Upload PDF
     ↓
Text Extraction
     ↓
Text Cleaning
     ↓
Chunking
     ↓
Embedding Generation
     ↓
FAISS Vector Index
     ↓
Semantic Search
     ↓
Relevant Document Context
     ↓
RAG / Generative AI Layer
