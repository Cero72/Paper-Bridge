#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import numpy as np
import argparse
import networkx as nx
import pickle
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional, Set
import datetime
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
DATA_DIR = Path("data")
GRAPH_DIR = Path("data")
RESULTS_DIR = Path("data/results")
S2_DATA_CACHE = Path("data/s2_cache")

def ensure_dirs():
    """Ensure necessary directories exist."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    GRAPH_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    S2_DATA_CACHE.mkdir(parents=True, exist_ok=True)

def load_graph(filename: str = "paper_graph.gpickle") -> nx.Graph:
    """Load graph from file."""
    filepath = GRAPH_DIR / filename
    
    if not filepath.exists():
        raise FileNotFoundError(f"Graph file not found: {filepath}")
    
    with open(filepath, 'rb') as f:
        G = pickle.load(f)
    
    logger.info(f"Loaded graph with {len(G.nodes)} nodes and {len(G.edges)} edges")
    return G

def load_papers(filename: str = "papers.json") -> Dict[str, Any]:
    """Load papers from JSON file."""
    filepath = Path("data/raw") / filename
    
    if not filepath.exists():
        raise FileNotFoundError(f"Papers file not found: {filepath}")
    
    with open(filepath, 'r', encoding='utf-8') as f:
        papers = json.load(f)
    
    logger.info(f"Loaded {len(papers)} papers from {filepath}")
    return papers

def load_semantic_scholar_data(paper_id: str) -> Dict[str, Any]:
    """
    Load Semantic Scholar data for a paper, either from cache or fetch it.
    
    Args:
        paper_id: ArXiv paper ID
    
    Returns:
        Dict with Semantic Scholar data or empty dict if not available
    """
    # Check cache first
    cache_file = S2_DATA_CACHE / f"{paper_id}.json"
    
    if cache_file.exists():
        try:
            with open(cache_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Error loading cached S2 data for {paper_id}: {e}")
    
    # If not in cache, try to fetch it
    try:
        # Import the fetch function
        from ingest import fetch_semantic_scholar
        
        # Fetch the data
        s2_data = fetch_semantic_scholar(paper_id)
        
        # Cache the result if we got something
        if s2_data:
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(s2_data, f, indent=2)
            
            return s2_data
        
    except Exception as e:
        logger.warning(f"Error fetching S2 data for {paper_id}: {e}")
    
    # Return empty dict if we failed
    return {}

def years_old(paper_id: str, papers: Dict[str, Any]) -> float:
    """Calculate how many years old a paper is."""
    if paper_id not in papers:
        return 0.0
    
    paper = papers[paper_id]
    year_str = paper.get('year', 'Unknown')
    
    if year_str == 'Unknown':
        return 0.0
    
    try:
        year = int(year_str)
        current_year = datetime.datetime.now().year
        return current_year - year
    except ValueError:
        return 0.0

def get_citation_score(paper_id: str, papers: Dict[str, Any]) -> float:
    """
    Calculate citation score based on Semantic Scholar data.
    
    Args:
        paper_id: ArXiv paper ID
        papers: Dict mapping paper IDs to metadata
    
    Returns:
        Citation score, normalized between 0 and 1
    """
    # Load S2 data
    s2_data = load_semantic_scholar_data(paper_id)
    
    # Get citation count from S2 data
    citation_count = s2_data.get('citationCount', 0)
    
    # Simple normalization - log scale to handle papers with many citations
    if citation_count > 0:
        return min(1.0, np.log1p(citation_count) / 10.0)  # Cap at 1.0, divide by 10 to normalize
    
    return 0.0

def get_reference_overlap(paper_id: str, topic_a: str, topic_b: str, G: nx.Graph, papers: Dict[str, Any]) -> float:
    """
    Calculate reference overlap between a paper and the papers in two topics.
    
    Args:
        paper_id: ArXiv paper ID
        topic_a: First topic node ID
        topic_b: Second topic node ID
        G: NetworkX graph
        papers: Dict mapping paper IDs to metadata
    
    Returns:
        Reference overlap score between 0 and 1
    """
    # Get papers in topic A and B
    topic_a_papers = set()
    topic_b_papers = set()
    
    for paper, attrs in G.nodes(data=True):
        if attrs.get('type') == 'paper':
            # Check if this paper is connected to topic_a
            if G.has_edge(paper, topic_a):
                topic_a_papers.add(paper)
            # Check if this paper is connected to topic_b
            if G.has_edge(paper, topic_b):
                topic_b_papers.add(paper)
    
    # Get S2 data for the paper
    s2_data = load_semantic_scholar_data(paper_id)
    
    # Get references from S2 data
    references: Set[str] = set()
    citations: Set[str] = set()
    
    # Extract references
    if 'related' in s2_data and 'references' in s2_data['related']:
        for ref in s2_data['related']['references']:
            if 'arxivId' in ref and ref['arxivId']:
                references.add(ref['arxivId'])
    
    # Extract citations
    if 'related' in s2_data and 'citations' in s2_data['related']:
        for cit in s2_data['related']['citations']:
            if 'arxivId' in cit and cit['arxivId']:
                citations.add(cit['arxivId'])
    
    # Calculate overlap with topic A and B
    ref_overlap_a = len(references.intersection(topic_a_papers))
    ref_overlap_b = len(references.intersection(topic_b_papers))
    
    cit_overlap_a = len(citations.intersection(topic_a_papers))
    cit_overlap_b = len(citations.intersection(topic_b_papers))
    
    # Combine overlaps - more weight to references than citations
    combined_overlap_a = ref_overlap_a * 2 + cit_overlap_a
    combined_overlap_b = ref_overlap_b * 2 + cit_overlap_b
    
    # Calculate final score
    if len(topic_a_papers) == 0 or len(topic_b_papers) == 0:
        return 0.0
    
    # Normalize by the size of the topic paper sets
    norm_overlap_a = combined_overlap_a / len(topic_a_papers)
    norm_overlap_b = combined_overlap_b / len(topic_b_papers)
    
    # Final score - use harmonic mean for balance
    if norm_overlap_a == 0 or norm_overlap_b == 0:
        return 0.0
    
    return 2 * (norm_overlap_a * norm_overlap_b) / (norm_overlap_a + norm_overlap_b)

def bridge_score(paper_id: str, topic_a: str, topic_b: str, G: nx.Graph, papers: Dict[str, Any],
               alpha: float = 1.0, beta: float = 0.3, gamma: float = 0.1, delta: float = 0.5, epsilon: float = 0.4) -> float:
    """
    Calculate enhanced bridge score for a paper between two topics using Semantic Scholar data.
    
    Args:
        paper_id: Paper ID
        topic_a: First topic node ID
        topic_b: Second topic node ID
        G: NetworkX graph
        papers: Dict mapping paper IDs to metadata
        alpha: Weight for inverse distance
        beta: Weight for centrality
        gamma: Weight for age penalty (older papers get penalized)
        delta: Weight for citation count (more cited papers get boosted)
        epsilon: Weight for reference overlap with topic papers
        
    Returns:
        Bridge score
    """
    # Check if paper and topics exist in graph
    if paper_id not in G or topic_a not in G or topic_b not in G:
        return 0.0
    
    try:
        # Try to find shortest paths considering edge weights
        # Lower weight = stronger connection
        try:
            # Use 'weight' attribute as the edge cost
            path_a = nx.shortest_path(G, paper_id, topic_a, weight='weight')
            dist_a = len(path_a) - 1  # number of edges in the path
        except (nx.NetworkXNoPath, nx.NetworkXError):
            # If no direct path exists, use the paper's topic as an intermediary
            # Find the paper's assigned topic
            paper_topics = [n for n in G.neighbors(paper_id) if n.startswith('topic_')]
            if not paper_topics:
                return 0.0  # Paper not connected to any topic
                
            # Use the first topic (there should be only one)
            paper_topic = paper_topics[0]
            
            # Distance from paper to its topic
            paper_to_topic_dist = 1
            
            # Distance from paper's topic to target topic (if they're different)
            if paper_topic == topic_a:
                topic_to_topic_a_dist = 0
            else:
                try:
                    path = nx.shortest_path(G, paper_topic, topic_a)
                    topic_to_topic_a_dist = len(path) - 1
                except (nx.NetworkXNoPath, nx.NetworkXError):
                    topic_to_topic_a_dist = float('inf')
                    
            dist_a = paper_to_topic_dist + topic_to_topic_a_dist
                
        try:
            # Repeat for topic B
            path_b = nx.shortest_path(G, paper_id, topic_b, weight='weight')
            dist_b = len(path_b) - 1
        except (nx.NetworkXNoPath, nx.NetworkXError):
            paper_topics = [n for n in G.neighbors(paper_id) if n.startswith('topic_')]
            if not paper_topics:
                return 0.0
                
            paper_topic = paper_topics[0]
            paper_to_topic_dist = 1
            
            if paper_topic == topic_b:
                topic_to_topic_b_dist = 0
            else:
                try:
                    path = nx.shortest_path(G, paper_topic, topic_b)
                    topic_to_topic_b_dist = len(path) - 1
                except (nx.NetworkXNoPath, nx.NetworkXError):
                    topic_to_topic_b_dist = float('inf')
                    
            dist_b = paper_to_topic_dist + topic_to_topic_b_dist
        
        # If either distance is infinite, the paper doesn't bridge the topics
        if dist_a == float('inf') or dist_b == float('inf'):
            return 0.0
            
        # Distance component - papers closer to both topics get higher scores
        distance_score = 1.0 / (1.0 + dist_a + dist_b)  # Add 1 to avoid division by zero
        
        # Centrality component - more connected papers get higher scores
        centrality = nx.degree_centrality(G)[paper_id]
        
        # Age component - newer papers get higher scores (less penalty)
        age = years_old(paper_id, papers)
        age_penalty = age / 10.0  # Normalize age penalty
        
        # Citation component - more cited papers get higher scores
        citation_score = get_citation_score(paper_id, papers)
        
        # Reference overlap component - papers that cite both topics get higher scores
        reference_overlap_score = get_reference_overlap(paper_id, topic_a, topic_b, G, papers)
        
        # Combine into final score
        score = (
            alpha * distance_score + 
            beta * centrality - 
            gamma * age_penalty + 
            delta * citation_score + 
            epsilon * reference_overlap_score
        )
        
        return score
        
    except Exception as e:
        # Log the error and return 0 for this paper
        logger.error(f"Error calculating bridge score for {paper_id}: {e}")
        return 0.0

def find_bridge_papers(topic_a: str, topic_b: str, G: nx.Graph, papers: Dict[str, Any],
                      top_k: int = 5, alpha: float = 1.0, beta: float = 0.3, gamma: float = 0.1) -> List[Tuple[str, float]]:
    """
    Find papers that best bridge between two topics.
    
    Args:
        topic_a: First topic node ID
        topic_b: Second topic node ID
        G: NetworkX graph
        papers: Dict mapping paper IDs to metadata
        top_k: Number of top papers to return
        alpha: Weight for inverse distance
        beta: Weight for centrality
        gamma: Weight for recency
        
    Returns:
        List of (paper_id, score) tuples
    """
    # Check if topics exist in graph
    if topic_a not in G or topic_b not in G:
        print(f"One or both topics not found in graph: {topic_a}, {topic_b}")
        return []
    
    # Get all paper nodes
    paper_nodes = [n for n, d in G.nodes(data=True) if d.get('type') == 'paper']
    
    # Calculate bridge score for each paper
    scores = []
    for paper_id in paper_nodes:
        score = bridge_score(paper_id, topic_a, topic_b, G, papers, alpha, beta, gamma)
        scores.append((paper_id, score))
    
    # Remove papers with zero scores
    scores = [(pid, score) for pid, score in scores if score > 0]
    
    # If no papers have positive scores, try papers directly connected to topics
    if not scores:
        print("No papers with positive bridge scores found. Finding papers directly connected to topics...")
        
        # Get papers directly connected to each topic
        papers_topic_a = set([n for n in G.neighbors(topic_a) if not n.startswith('topic_')])
        papers_topic_b = set([n for n in G.neighbors(topic_b) if not n.startswith('topic_')])
        
        # Prioritize papers that are in one topic but close to the other
        # or papers with cross-topic connections
        for paper_id in papers_topic_a.union(papers_topic_b):
            age_score = 0.1 * (10 - min(10, years_old(paper_id, papers)))
            centrality = nx.degree_centrality(G)[paper_id]
            
            # Higher score if connected to both topics or has many connections
            if paper_id in papers_topic_a and paper_id in papers_topic_b:
                score = 0.9 + centrality + age_score  # Directly bridges both topics
            elif paper_id in papers_topic_a:
                score = 0.5 + centrality + age_score
            else:
                score = 0.4 + centrality + age_score
                
            scores.append((paper_id, score))
    
    # Sort by score (descending) and take top_k
    scores.sort(key=lambda x: x[1], reverse=True)
    top_papers = scores[:top_k]
    
    return top_papers

def explain_bridge(paper_id: str, topic_a: str, topic_b: str, G: nx.Graph, papers: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate explanation for why a paper is a good bridge.
    
    Args:
        paper_id: Paper ID
        topic_a: First topic node ID
        topic_b: Second topic node ID
        G: NetworkX graph
        papers: Dict mapping paper IDs to metadata
        
    Returns:
        Dict with explanation details
    """
    # Basic paper information
    explanation = {
        "paper_id": paper_id,
        "title": papers.get(paper_id, {}).get("title", "Unknown"),
        "year": papers.get(paper_id, {}).get("year", "Unknown"),
        "path_to_topic_a": None,
        "path_to_topic_b": None,
        "centrality": nx.degree_centrality(G)[paper_id] if paper_id in G else 0,
        "age": years_old(paper_id, papers),
        "connection_strength": 0.0,
        "num_connections": 0,
        "bridge_type": "unknown"
    }
    
    # Find paths to topics
    try:
        path_a = nx.shortest_path(G, paper_id, topic_a, weight='weight')
        explanation["path_to_topic_a"] = path_a
        
        # Add path explanation
        path_details = []
        for i in range(len(path_a) - 1):
            node1, node2 = path_a[i], path_a[i+1]
            edge_type = G.edges[node1, node2].get('type', 'connected')
            path_details.append(f"{node1} --({edge_type})--> {node2}")
        explanation["path_a_details"] = path_details
    except (nx.NetworkXNoPath, nx.NetworkXError):
        # No direct path found, try finding through the paper's topic
        paper_topics = [n for n in G.neighbors(paper_id) if n.startswith('topic_')]
        if paper_topics:
            paper_topic = paper_topics[0]
            explanation["paper_topic"] = paper_topic
            
            if paper_topic == topic_a:
                explanation["path_to_topic_a"] = [paper_id, topic_a]
            else:
                try:
                    topic_path = nx.shortest_path(G, paper_topic, topic_a)
                    explanation["path_to_topic_a"] = [paper_id, paper_topic] + topic_path[1:]
                except (nx.NetworkXNoPath, nx.NetworkXError):
                    pass
    
    # Repeat for topic B
    try:
        path_b = nx.shortest_path(G, paper_id, topic_b, weight='weight')
        explanation["path_to_topic_b"] = path_b
        
        # Add path explanation
        path_details = []
        for i in range(len(path_b) - 1):
            node1, node2 = path_b[i], path_b[i+1]
            edge_type = G.edges[node1, node2].get('type', 'connected')
            path_details.append(f"{node1} --({edge_type})--> {node2}")
        explanation["path_b_details"] = path_details
    except (nx.NetworkXNoPath, nx.NetworkXError):
        paper_topics = [n for n in G.neighbors(paper_id) if n.startswith('topic_')]
        if paper_topics:
            paper_topic = paper_topics[0]
            explanation["paper_topic"] = paper_topic
            
            if paper_topic == topic_b:
                explanation["path_to_topic_b"] = [paper_id, topic_b]
            else:
                try:
                    topic_path = nx.shortest_path(G, paper_topic, topic_b)
                    explanation["path_to_topic_b"] = [paper_id, paper_topic] + topic_path[1:]
                except (nx.NetworkXNoPath, nx.NetworkXError):
                    pass
    
    # Calculate connection strength based on bridges
    explanation["num_connections"] = G.degree(paper_id)
    
    # Determine bridge type
    if explanation["path_to_topic_a"] and explanation["path_to_topic_b"]:
        # Check if paper directly connects to both topics
        if len(explanation["path_to_topic_a"]) == 2 and len(explanation["path_to_topic_b"]) == 2:
            explanation["bridge_type"] = "direct_bridge"
            explanation["connection_strength"] = 1.0
        # Check if paper is in one topic but has direct connections to papers in the other
        elif len(explanation["path_to_topic_a"]) == 2 or len(explanation["path_to_topic_b"]) == 2:
            explanation["bridge_type"] = "semi_direct_bridge"
            explanation["connection_strength"] = 0.7
        else:
            explanation["bridge_type"] = "indirect_bridge"
            explanation["connection_strength"] = 0.4
    
    # Extract keywords from abstract for context
    if paper_id in papers and 'abstract' in papers[paper_id]:
        abstract = papers[paper_id]['abstract']
        # Simple keyword extraction - split and keep longer words
        words = abstract.lower().split()
        keywords = [w for w in words if len(w) > 5 and w.isalpha()]
        # Count frequency
        from collections import Counter
        word_counts = Counter(keywords)
        # Get top 10 keywords
        explanation["keywords"] = [word for word, count in word_counts.most_common(10)]
    
    return explanation

def save_results(results: List[Tuple[str, float]], explanations: List[Dict[str, Any]], 
                topic_a: str, topic_b: str, filename: str = "bridge_results.json"):
    """Save bridge paper results to file."""
    ensure_dirs()
    filepath = RESULTS_DIR / filename
    
    # Format results for saving
    output = {
        "topic_a": topic_a,
        "topic_b": topic_b,
        "timestamp": datetime.datetime.now().isoformat(),
        "results": [
            {
                "paper_id": paper_id,
                "score": score,
                "explanation": explanation
            }
            for (paper_id, score), explanation in zip(results, explanations)
        ]
    }
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2)
    
    print(f"Saved bridge results to {filepath}")

def main():
    parser = argparse.ArgumentParser(description="Find papers that bridge between topics")
    parser.add_argument("--graph", default="paper_graph.gpickle", help="Input graph filename")
    parser.add_argument("--papers", default="papers.json", help="Input papers JSON filename")
    parser.add_argument("--topic-a", required=True, help="First topic ID (e.g., 'topic_3')")
    parser.add_argument("--topic-b", required=True, help="Second topic ID (e.g., 'topic_7')")
    parser.add_argument("--top-k", type=int, default=5, help="Number of top papers to return")
    parser.add_argument("--alpha", type=float, default=1.0, help="Weight for inverse distance")
    parser.add_argument("--beta", type=float, default=0.3, help="Weight for centrality")
    parser.add_argument("--gamma", type=float, default=0.1, help="Weight for recency")
    parser.add_argument("--output", default="bridge_results.json", help="Output results filename")
    
    args = parser.parse_args()
    
    # Format topic IDs if integer was provided
    topic_a = args.topic_a if args.topic_a.startswith('topic_') else f"topic_{args.topic_a}"
    topic_b = args.topic_b if args.topic_b.startswith('topic_') else f"topic_{args.topic_b}"
    
    G = load_graph(args.graph)
    papers = load_papers(args.papers)
    
    # Find bridge papers
    bridge_papers = find_bridge_papers(topic_a, topic_b, G, papers, args.top_k, args.alpha, args.beta, args.gamma)
    
    # Generate explanations
    explanations = [explain_bridge(paper_id, topic_a, topic_b, G, papers) for paper_id, _ in bridge_papers]
    
    # Print results
    print(f"\nTop {len(bridge_papers)} bridge papers between {topic_a} and {topic_b}:")
    for i, (paper_id, score) in enumerate(bridge_papers, 1):
        title = papers.get(paper_id, {}).get("title", "Unknown")
        year = papers.get(paper_id, {}).get("year", "Unknown")
        print(f"[{i}] {title} ({year}) - Score: {score:.4f}")
    
    # Save results
    save_results(bridge_papers, explanations, topic_a, topic_b, args.output)

if __name__ == "__main__":
    main() 