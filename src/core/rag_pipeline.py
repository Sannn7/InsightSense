import os
import numpy as np

# 1. Loaders
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 2. Vector Store (FAISS)
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# 3. DIRECT LIBRARIES (Bypassing LangChain wrappers)
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

class RAGPipeline:
    def __init__(self):
        # Embeddings for Vector Search
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        
        # Cross Encoder for Reranking (Directly from sentence-transformers)
        self.cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        
        self.vector_store = None
        self.bm25 = None
        self.chunks = [] # We keep chunks in memory for BM25 lookup

    def process_pdf(self, file_path):
        """
        Manually builds a Hybrid Search (FAISS + BM25) pipeline.
        """
        # 1. Load PDF
        try:
            loader = PyPDFLoader(file_path)
            docs = loader.load()
        except Exception as e:
            return f"Error loading PDF: {str(e)}"
        
        if not docs:
            return "PDF was empty."

        # 2. Split Text
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        self.chunks = text_splitter.split_documents(docs)

        # 3. Build Vector Index (FAISS)
        self.vector_store = FAISS.from_documents(self.chunks, self.embeddings)

        # 4. Build Keyword Index (BM25) manually
        # We simple tokenize by splitting on spaces (lowercased)
        tokenized_corpus = [doc.page_content.lower().split() for doc in self.chunks]
        self.bm25 = BM25Okapi(tokenized_corpus)
        
        return f"Custom Hybrid Pipeline Built: {len(self.chunks)} chunks indexed."

    def get_context(self, query):
        """
        Performs Hybrid Search (Vector + Keyword) and Reranking manually.
        """
        if not self.vector_store or not self.bm25:
            return []
        
        # --- Step A: Get Candidates from Vector Search ---
        # Get top 5 semantic matches
        vector_results = self.vector_store.similarity_search(query, k=5)
        
        # --- Step B: Get Candidates from Keyword Search (BM25) ---
        # Get top 5 keyword matches
        tokenized_query = query.lower().split()
        # BM25 returns the actual text chunks, so we map them back to Documents
        bm25_top_n = self.bm25.get_top_n(tokenized_query, self.chunks, n=5)
        
        # --- Step C: Combine & Deduplicate (Hybrid) ---
        # We use a dictionary to remove duplicates based on page content
        unique_docs = {}
        
        # Add Vector results
        for doc in vector_results:
            unique_docs[doc.page_content] = doc
            
        # Add BM25 results
        for doc in bm25_top_n:
            unique_docs[doc.page_content] = doc
            
        combined_docs = list(unique_docs.values())
        
        # --- Step D: Rerank (Cross-Encoder) ---
        # Prepare pairs: [[query, doc1], [query, doc2], ...]
        pairs = [[query, doc.page_content] for doc in combined_docs]
        
        # Get scores
        scores = self.cross_encoder.predict(pairs)
        
        # Sort documents by score (highest to lowest)
        # We zip docs and scores together, sort, and unzip
        scored_docs = sorted(zip(combined_docs, scores), key=lambda x: x[1], reverse=True)
        
        # Return top 3 reranked docs
        final_top_3 = [doc for doc, score in scored_docs[:3]]
        
        return final_top_3