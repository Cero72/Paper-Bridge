#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import numpy as np
import argparse
from pathlib import Path
from typing import Dict, List, Any, Tuple
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize

# Constants
DATA_DIR = Path("data")
TOPIC_DIR = Path("data")

def ensure_dirs():
    """Ensure necessary directories exist."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    TOPIC_DIR.mkdir(parents=True, exist_ok=True)

def load_papers(filename: str = "papers.json") -> Dict[str, Any]:
    """Load papers from JSON file."""
    filepath = Path("data/raw") / filename
    
    if not filepath.exists():
        raise FileNotFoundError(f"Papers file not found: {filepath}")
    
    with open(filepath, 'r', encoding='utf-8') as f:
        papers = json.load(f)
    
    print(f"Loaded {len(papers)} papers from {filepath}")
    return papers

def load_embeddings(filename: str = "embeddings.npz") -> Tuple[List[str], np.ndarray]:
    """Load embeddings from NPZ file."""
    filepath = DATA_DIR / filename
    
    if not filepath.exists():
        raise FileNotFoundError(f"Embeddings file not found: {filepath}")
    
    data = np.load(filepath, allow_pickle=True)
    paper_ids = data["ids"].tolist()
    embeddings = data["embeddings"]
    
    print(f"Loaded embeddings for {len(paper_ids)} papers with dimension {embeddings.shape[1]}")
    return paper_ids, embeddings

def create_topic_model(papers: Dict[str, Any], paper_ids: List[str], embeddings: np.ndarray, 
                       n_clusters: int = 5) -> Tuple[List[int], Dict[int, str]]:
    """
    Create a simple topic model using K-means clustering.
    
    Args:
        papers: Dict mapping paper IDs to metadata
        paper_ids: List of paper IDs corresponding to embeddings
        embeddings: NumPy array of embeddings
        n_clusters: Number of clusters to create
        
    Returns:
        Tuple of (cluster assignments, topic names)
    """
    # Normalize the embeddings
    normalized_embeddings = normalize(embeddings)
    
    # Apply K-means clustering
    print(f"Clustering embeddings into {n_clusters} topics...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    topics = kmeans.fit_predict(normalized_embeddings)
    
    # Get the centers of the clusters
    centers = kmeans.cluster_centers_
    
    # Get the papers closest to each cluster center
    topic_papers = {}
    for topic_id in range(n_clusters):
        mask = topics == topic_id
        if np.any(mask):
            topic_papers[topic_id] = []
            topic_indices = np.where(mask)[0]
            for idx in topic_indices:
                paper_id = paper_ids[idx]
                topic_papers[topic_id].append(paper_id)
    
    # Create topic names based on most frequent words in titles of each cluster
    topic_names = {}
    import re
    from collections import Counter
    
    # Stop words to exclude
    stop_words = {"a", "an", "the", "and", "or", "but", "is", "are", "in", "on", "at", "to", "for", "with", "by", "of"}
    
    for topic_id, cluster_paper_ids in topic_papers.items():
        # Collect all words from titles in this cluster
        all_words = []
        for paper_id in cluster_paper_ids:
            title = papers[paper_id]["title"]
            # Convert to lowercase and split on non-alphanumeric
            words = re.findall(r'\b[a-zA-Z]{3,}\b', title.lower())
            # Remove stop words
            words = [w for w in words if w not in stop_words]
            all_words.extend(words)
        
        # Count frequencies and get top words
        if all_words:
            counter = Counter(all_words)
            top_words = [word for word, _ in counter.most_common(3)]
            topic_names[topic_id] = " + ".join(top_words)
        else:
            topic_names[topic_id] = f"Topic {topic_id}"
    
    print(f"Created {len(topic_names)} topics")
    return topics, topic_names

def save_topic_model(topics: List[int], topic_names: Dict[int, str], paper_ids: List[str], 
                     mapping_name: str = "paper_topics.json"):
    """Save topic model and paper-topic mapping."""
    ensure_dirs()
    
    # Save paper-topic mapping
    mapping = {paper_ids[i]: int(topics[i]) for i in range(len(paper_ids))}
    
    mapping_path = TOPIC_DIR / mapping_name
    with open(mapping_path, 'w', encoding='utf-8') as f:
        json.dump(mapping, f, indent=2)
    
    print(f"Saved paper-topic mapping to {mapping_path}")
    
    # Save topic info as CSV for easy inspection
    topic_info = pd.DataFrame([
        {"Topic": topic_id, "Name": name, "Count": list(topics).count(topic_id)}
        for topic_id, name in topic_names.items()
    ])
    topic_info_path = TOPIC_DIR / "topic_info.csv"
    topic_info.to_csv(topic_info_path, index=False)
    print(f"Saved topic info to {topic_info_path}")
    
    # Also create a PCA visualization of the embeddings with cluster colors
    try:
        import matplotlib.pyplot as plt
        
        # Apply PCA to reduce to 2 dimensions
        pca = PCA(n_components=2)
        embeddings_2d = pca.fit_transform(normalize(np.vstack([emb for emb in np.load(DATA_DIR / "embeddings.npz")["embeddings"]])))
        
        # Create the plot
        plt.figure(figsize=(10, 8))
        for topic_id in range(max(topics) + 1):
            mask = topics == topic_id
            if np.any(mask):
                plt.scatter(
                    embeddings_2d[mask, 0], 
                    embeddings_2d[mask, 1],
                    label=f"Topic {topic_id}: {topic_names[topic_id]}"
                )
        
        plt.title("Paper Embeddings - PCA Projection")
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.legend(loc="best")
        plt.tight_layout()
        
        # Save the visualization
        vis_path = TOPIC_DIR / "topic_visualization.png"
        plt.savefig(vis_path)
        print(f"Saved topic visualization to {vis_path}")
    except Exception as e:
        print(f"Could not create visualization: {e}")

def main():
    parser = argparse.ArgumentParser(description="Create topic model from paper embeddings")
    parser.add_argument("--papers", default="papers.json", help="Input papers JSON filename")
    parser.add_argument("--embeddings", default="embeddings.npz", help="Input embeddings NPZ filename")
    parser.add_argument("--n-clusters", type=int, default=5, help="Number of clusters to create")
    parser.add_argument("--output", default="paper_topics.json", help="Output mapping filename")
    
    args = parser.parse_args()
    
    papers = load_papers(args.papers)
    paper_ids, embeddings = load_embeddings(args.embeddings)
    topics, topic_names = create_topic_model(papers, paper_ids, embeddings, args.n_clusters)
    save_topic_model(topics, topic_names, paper_ids, args.output)

if __name__ == "__main__":
    main() 