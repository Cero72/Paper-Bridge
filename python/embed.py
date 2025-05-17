#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import numpy as np
import argparse
from pathlib import Path
from typing import Dict, List, Any
from sentence_transformers import SentenceTransformer

# Constants
DATA_DIR = Path("data/raw")
EMBED_DIR = Path("data")

def ensure_dirs():
    """Ensure necessary directories exist."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    EMBED_DIR.mkdir(parents=True, exist_ok=True)

def load_papers(filename: str = "papers.json") -> Dict[str, Any]:
    """Load papers from JSON file."""
    filepath = DATA_DIR / filename
    
    if not filepath.exists():
        raise FileNotFoundError(f"Papers file not found: {filepath}")
    
    with open(filepath, 'r', encoding='utf-8') as f:
        papers = json.load(f)
    
    print(f"Loaded {len(papers)} papers from {filepath}")
    return papers

def generate_embeddings(papers: Dict[str, Any], model_name: str = "all-MiniLM-L6-v2") -> Dict[str, np.ndarray]:
    """
    Generate embeddings for paper abstracts using SentenceTransformer.
    
    Args:
        papers: Dict mapping paper IDs to metadata
        model_name: Name of the SentenceTransformer model to use
        
    Returns:
        Dict mapping paper IDs to embedding vectors
    """
    # Load model
    print(f"Loading SentenceTransformer model: {model_name}")
    model = SentenceTransformer(model_name)
    
    # Prepare abstracts
    paper_ids = list(papers.keys())
    abstracts = [papers[pid]["abstract"] for pid in paper_ids]
    
    # Generate embeddings
    print(f"Generating embeddings for {len(abstracts)} abstracts...")
    embeddings = model.encode(abstracts, show_progress_bar=True)
    
    # Map paper IDs to embeddings
    embedding_dict = {pid: embeddings[i] for i, pid in enumerate(paper_ids)}
    
    print(f"Generated {len(embedding_dict)} embeddings with dimension {embeddings.shape[1]}")
    return embedding_dict

def save_embeddings(embeddings: Dict[str, np.ndarray], filename: str = "embeddings.npz"):
    """Save embeddings to NPZ file."""
    ensure_dirs()
    filepath = EMBED_DIR / filename
    
    # Convert dict to arrays
    paper_ids = list(embeddings.keys())
    embedding_array = np.array([embeddings[pid] for pid in paper_ids])
    
    # Save to NPZ
    np.savez(filepath, ids=paper_ids, embeddings=embedding_array)
    
    print(f"Saved embeddings for {len(paper_ids)} papers to {filepath}")
    
    # Also save a mapping file for easy lookup
    map_filepath = EMBED_DIR / "embedding_map.json"
    with open(map_filepath, 'w', encoding='utf-8') as f:
        json.dump({"ids": paper_ids}, f)

def main():
    parser = argparse.ArgumentParser(description="Generate embeddings for paper abstracts")
    parser.add_argument("--input", default="papers.json", help="Input JSON filename")
    parser.add_argument("--model", default="all-MiniLM-L6-v2", help="SentenceTransformer model name")
    parser.add_argument("--output", default="embeddings.npz", help="Output NPZ filename")
    
    args = parser.parse_args()
    
    papers = load_papers(args.input)
    embeddings = generate_embeddings(papers, args.model)
    save_embeddings(embeddings, args.output)

if __name__ == "__main__":
    main() 