#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import numpy as np
import argparse
import networkx as nx
import pickle
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional

# Constants
DATA_DIR = Path("data")
GRAPH_DIR = Path("data")

def ensure_dirs():
    """Ensure necessary directories exist."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    GRAPH_DIR.mkdir(parents=True, exist_ok=True)

def load_papers(filename: str = "papers.json") -> Dict[str, Any]:
    """Load papers from JSON file."""
    filepath = Path("data/raw") / filename
    
    if not filepath.exists():
        raise FileNotFoundError(f"Papers file not found: {filepath}")
    
    with open(filepath, 'r', encoding='utf-8') as f:
        papers = json.load(f)
    
    print(f"Loaded {len(papers)} papers from {filepath}")
    return papers

def load_paper_topics(filename: str = "paper_topics.json") -> Dict[str, int]:
    """Load paper-topic mapping from JSON file."""
    filepath = DATA_DIR / filename
    
    if not filepath.exists():
        raise FileNotFoundError(f"Paper-topic mapping file not found: {filepath}")
    
    with open(filepath, 'r', encoding='utf-8') as f:
        mapping = json.load(f)
    
    print(f"Loaded topic mapping for {len(mapping)} papers")
    return mapping

def load_topic_info(filename: str = "topic_info.csv") -> Dict[int, str]:
    """Load topic info from CSV file."""
    import pandas as pd
    
    filepath = DATA_DIR / filename
    
    if not filepath.exists():
        raise FileNotFoundError(f"Topic info file not found: {filepath}")
    
    df = pd.read_csv(filepath)
    topic_dict = {row["Topic"]: row["Name"] for _, row in df.iterrows() if row["Topic"] != -1}
    
    print(f"Loaded info for {len(topic_dict)} topics")
    return topic_dict

def build_graph(papers: Dict[str, Any], paper_topics: Dict[str, int], topic_info: Dict[int, str]) -> nx.Graph:
    """
    Build a knowledge graph of papers and topics.
    
    Args:
        papers: Dict mapping paper IDs to metadata
        paper_topics: Dict mapping paper IDs to topic IDs
        topic_info: Dict mapping topic IDs to topic names
        
    Returns:
        NetworkX graph
    """
    # Create graph
    G = nx.Graph()
    
    # Add paper nodes
    for paper_id, paper_data in papers.items():
        G.add_node(paper_id, 
                   type='paper',
                   title=paper_data.get('title', 'Unknown'),
                   year=paper_data.get('year', 'Unknown'),
                   authors=paper_data.get('authors', []),
                   abstract=paper_data.get('abstract', ''))
    
    # Add topic nodes
    for topic_id, topic_name in topic_info.items():
        G.add_node(f"topic_{topic_id}", type='topic', name=topic_name)
    
    # Add paper-topic edges
    topic_papers = {}  # Dictionary to group papers by topic
    for paper_id, topic_id in paper_topics.items():
        if topic_id != -1:  # Skip outliers
            topic_key = f"topic_{topic_id}"
            G.add_edge(paper_id, topic_key, type='belongs_to', weight=1.0)
            
            # Group papers by topic for citation simulation
            if topic_key not in topic_papers:
                topic_papers[topic_key] = []
            topic_papers[topic_key].append(paper_id)
    
    # Add citation edges (paper-paper)
    citation_count = 0
    for paper_id, paper_data in papers.items():
        # Try to use real citations if available
        citations_added = False
        if 'related' in paper_data:
            if 'references' in paper_data['related']:
                for ref in paper_data['related']['references']:
                    if 'arxivId' in ref and ref['arxivId'] and ref['arxivId'] in papers:
                        G.add_edge(paper_id, ref['arxivId'], type='cites', weight=0.7)
                        citation_count += 1
                        citations_added = True
                        
        # If no real citations found, add edges to a few papers in the same topic
        if not citations_added and paper_id in paper_topics:
            topic_id = paper_topics[paper_id]
            topic_key = f"topic_{topic_id}"
            if topic_key in topic_papers:
                same_topic_papers = [p for p in topic_papers[topic_key] if p != paper_id]
                # Add edges to up to 3 random papers in the same topic
                import random
                if same_topic_papers:
                    for similar_paper in random.sample(same_topic_papers, min(3, len(same_topic_papers))):
                        G.add_edge(paper_id, similar_paper, type='similar_topic', weight=0.5)
                        citation_count += 1
    
    # Add cross-topic connections between similar papers
    import random
    for topic_a, papers_a in topic_papers.items():
        for topic_b, papers_b in topic_papers.items():
            if topic_a != topic_b:
                # For each pair of topics, add a few cross-topic connections
                if papers_a and papers_b:
                    # Select up to 2 papers from each topic
                    papers_from_a = random.sample(papers_a, min(2, len(papers_a)))
                    papers_from_b = random.sample(papers_b, min(2, len(papers_b)))
                    
                    # Connect them
                    for paper_a in papers_from_a:
                        for paper_b in papers_from_b:
                            G.add_edge(paper_a, paper_b, type='cross_topic', weight=0.3)
                            citation_count += 1
    
    print(f"Built graph with {len(G.nodes)} nodes and {len(G.edges)} edges")
    print(f"  - {sum(1 for n, d in G.nodes(data=True) if d.get('type') == 'paper')} paper nodes")
    print(f"  - {sum(1 for n, d in G.nodes(data=True) if d.get('type') == 'topic')} topic nodes")
    print(f"  - {sum(1 for _, _, d in G.edges(data=True) if d.get('type') == 'belongs_to')} paper-topic edges")
    print(f"  - {citation_count} citation/similarity edges")
    
    return G

def save_graph(G: nx.Graph, filename: str = "paper_graph.gpickle"):
    """Save graph to file."""
    ensure_dirs()
    filepath = GRAPH_DIR / filename
    
    # Use pickle to save the graph instead of write_gpickle
    with open(filepath, 'wb') as f:
        pickle.dump(G, f)
    
    print(f"Saved graph to {filepath}")
    
    # Also save as GraphML for compatibility
    graphml_path = GRAPH_DIR / "paper_graph.graphml"
    try:
        nx.write_graphml(G, graphml_path)
        print(f"Saved graph in GraphML format to {graphml_path}")
    except Exception as e:
        print(f"Could not save as GraphML: {e}")

def main():
    parser = argparse.ArgumentParser(description="Build knowledge graph of papers and topics")
    parser.add_argument("--papers", default="papers.json", help="Input papers JSON filename")
    parser.add_argument("--topics", default="paper_topics.json", help="Input paper-topic mapping filename")
    parser.add_argument("--topic-info", default="topic_info.csv", help="Input topic info CSV filename")
    parser.add_argument("--output", default="paper_graph.gpickle", help="Output graph filename")
    
    args = parser.parse_args()
    
    papers = load_papers(args.papers)
    paper_topics = load_paper_topics(args.topics)
    topic_info = load_topic_info(args.topic_info)
    
    G = build_graph(papers, paper_topics, topic_info)
    save_graph(G, args.output)

if __name__ == "__main__":
    main() 