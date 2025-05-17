#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import pickle
import argparse
import networkx as nx
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import numpy as np
from pyvis.network import Network

# Constants
DATA_DIR = Path("data")
GRAPH_DIR = Path("data")
VIS_DIR = Path("data/visualizations")

def ensure_dirs():
    """Ensure necessary directories exist."""
    VIS_DIR.mkdir(parents=True, exist_ok=True)

def load_graph(filename="paper_graph.gpickle"):
    """Load the paper graph."""
    graph_path = GRAPH_DIR / filename
    if not graph_path.exists():
        raise FileNotFoundError(f"Graph file not found: {graph_path}")
    
    with open(graph_path, 'rb') as f:
        G = pickle.load(f)
    
    print(f"Loaded graph with {len(G.nodes)} nodes and {len(G.edges)} edges")
    return G

def load_papers(filename="papers.json"):
    """Load paper data."""
    papers_path = Path("data/raw") / filename
    if not papers_path.exists():
        raise FileNotFoundError(f"Papers file not found: {papers_path}")
    
    with open(papers_path, 'r', encoding='utf-8') as f:
        papers = json.load(f)
    
    print(f"Loaded {len(papers)} papers")
    return papers

def load_topics():
    """Load topic info."""
    topic_info_path = DATA_DIR / "topic_info.csv"
    if not topic_info_path.exists():
        raise FileNotFoundError(f"Topic info file not found: {topic_info_path}")
    
    df = pd.read_csv(topic_info_path)
    # Create a dict of topic_id -> topic_name
    topic_dict = {f"topic_{row['Topic']}": row['Name'] for _, row in df.iterrows() if row['Topic'] != -1}
    
    print(f"Loaded info for {len(topic_dict)} topics")
    return topic_dict

def load_bridge_papers():
    """Load bridge paper rankings if available."""
    bridge_results_path = DATA_DIR / "results" / "bridge_results.json"
    if not bridge_results_path.exists():
        return {}
    
    with open(bridge_results_path, 'r', encoding='utf-8') as f:
        results = json.load(f)
    
    # Convert to dict for easy lookup
    bridge_papers = {paper["id"]: paper["score"] for paper in results.get("papers", [])}
    return bridge_papers

def create_static_visualization(G, papers, topics, output="graph_visualization.png"):
    """Create a static visualization of the graph."""
    # Create a spring layout
    pos = nx.spring_layout(G, k=0.8, iterations=100, seed=42)
    
    plt.figure(figsize=(20, 16))
    
    # Define node types and colors
    node_types = {}
    for node in G.nodes():
        if node.startswith("topic_"):
            node_types[node] = "topic"
        else:
            node_types[node] = "paper"
    
    # Get unique colors for each topic
    topic_ids = [node for node in G.nodes() if node.startswith("topic_")]
    colors = cm.rainbow(np.linspace(0, 1, len(topic_ids)))
    topic_colors = {topic: colors[i] for i, topic in enumerate(topic_ids)}
    
    # Draw topic nodes
    topic_nodes = [node for node in G.nodes() if node_types[node] == "topic"]
    nx.draw_networkx_nodes(G, pos, nodelist=topic_nodes, node_size=1500, 
                           node_color=[topic_colors[node] for node in topic_nodes], alpha=0.8)
    
    # Load bridge papers if available
    bridge_papers = load_bridge_papers()
    
    # Draw paper nodes, colored by their primary topic
    paper_nodes = [node for node in G.nodes() if node_types[node] == "paper"]
    paper_colors = []
    paper_sizes = []
    
    for paper in paper_nodes:
        # Find connected topics
        connected_topics = [n for n in G.neighbors(paper) if n.startswith("topic_")]
        
        # Determine color
        if connected_topics:
            paper_colors.append(topic_colors[connected_topics[0]])
        else:
            paper_colors.append('gray')
        
        # Determine size (larger for bridge papers)
        if paper in bridge_papers:
            # Size based on bridge score (normalized between 300-800)
            min_size, max_size = 300, 800
            score = bridge_papers[paper]
            size = min_size + (max_size - min_size) * score
            paper_sizes.append(size)
        else:
            paper_sizes.append(300)
    
    nx.draw_networkx_nodes(G, pos, nodelist=paper_nodes, node_size=paper_sizes, 
                           node_color=paper_colors, alpha=0.6)
    
    # Draw edges
    nx.draw_networkx_edges(G, pos, alpha=0.2, width=0.5)
    
    # Draw topic labels
    topic_labels = {topic: topics.get(topic, topic) for topic in topic_nodes}
    nx.draw_networkx_labels(G, pos, labels=topic_labels, font_size=12, font_weight='bold')
    
    # Draw paper labels for bridge papers and important papers
    paper_labels = {}
    
    # First add bridge papers (if any)
    bridge_paper_ids = list(bridge_papers.keys())
    for paper in bridge_paper_ids:
        if paper in papers:
            paper_labels[paper] = papers[paper].get("title", "")[:30] + "..."
    
    # Then add some additional papers to fill up to 25 total
    remaining_slots = 25 - len(bridge_paper_ids)
    if remaining_slots > 0:
        for paper in paper_nodes[:remaining_slots]:
            if paper not in paper_labels and paper in papers:
                paper_labels[paper] = papers[paper].get("title", "")[:30] + "..."
    
    nx.draw_networkx_labels(G, pos, labels=paper_labels, font_size=8)
    
    plt.title("Paper-Topic Knowledge Graph", fontsize=20)
    plt.axis('off')
    plt.tight_layout()
    
    # Save visualization
    output_path = VIS_DIR / output
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved static graph visualization to {output_path}")
    plt.close()

def create_interactive_visualization(G, papers, topics, output="interactive_graph.html"):
    """Create an interactive visualization of the graph using pyvis."""
    # Create a pyvis network
    net = Network(height="800px", width="100%", notebook=False, directed=False, 
                  bgcolor="#ffffff", font_color="#333333")
    
    # Set physics options for better layout
    net.barnes_hut(gravity=-80000, central_gravity=0.3, spring_length=250, spring_strength=0.001, damping=0.09)
    
    # Get unique colors for each topic
    topic_ids = [node for node in G.nodes() if node.startswith("topic_")]
    colors = [f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}" for r, g, b, _ in 
              cm.rainbow(np.linspace(0, 1, len(topic_ids)))]
    topic_colors = {topic: colors[i] for i, topic in enumerate(topic_ids)}
    
    # Load bridge papers if available
    bridge_papers = load_bridge_papers()
    
    # Add topic nodes
    for topic_id in topic_ids:
        topic_name = topics.get(topic_id, topic_id)
        net.add_node(topic_id, label=topic_name, title=topic_name, size=30, 
                     color=topic_colors[topic_id], shape='diamond', font={"size": 20})
    
    # Add paper nodes
    for node in G.nodes():
        if not node.startswith("topic_"):
            # Skip if paper isn't in our database
            if node not in papers:
                continue
                
            # Find connected topics
            connected_topics = [n for n in G.neighbors(node) if n.startswith("topic_")]
            node_color = 'gray'
            if connected_topics:
                node_color = topic_colors[connected_topics[0]]
            
            # Create node tooltip with paper details
            paper = papers[node]
            title = paper.get("title", "Unknown Title")
            authors = ", ".join(paper.get("authors", [])[:3])
            if len(paper.get("authors", [])) > 3:
                authors += "..."
            year = paper.get("year", "Unknown Year")
            url = paper.get("url", f"https://arxiv.org/abs/{node}")
            
            # Create a rich HTML tooltip
            tooltip = f"""
            <div style='max-width:400px;'>
                <h3>{title}</h3>
                <p><strong>Authors:</strong> {authors}</p>
                <p><strong>Year:</strong> {year}</p>
                <p><strong>ID:</strong> {node}</p>
                <p><a href='{url}' target='_blank'>View paper</a></p>
            </div>
            """
            
            # Determine node size and label
            size = 15
            is_bridge = False
            
            # Check if it's a bridge paper and set size accordingly
            if node in bridge_papers:
                # Size based on bridge score (normalized between 15-30)
                min_size, max_size = 15, 30
                score = bridge_papers[node]
                size = min_size + (max_size - min_size) * score
                is_bridge = True
            
            # Always show labels for bridge papers, and some labels for regular papers
            show_label = is_bridge or len(title) < 40
            
            # Add badge for bridge papers
            if is_bridge:
                label = "🌉 " + title[:30] + "..."
                node_border = "#FF4500"  # Orange-red border for bridge papers
                border_width = 3
            else:
                label = title[:30] + "..." if show_label else ""
                node_border = node_color
                border_width = 1
            
            # Add paper node with enhanced styling
            net.add_node(node, label=label, title=tooltip, 
                         size=size, color={"background": node_color, "border": node_border},
                         borderWidth=border_width,
                         shape='dot', 
                         font={"size": 12, "face": "arial"},
                         url=url)  # Direct URL to paper
    
    # Add edges
    for edge in G.edges():
        # Skip self-loops
        if edge[0] == edge[1]:
            continue
            
        # Add edge with appropriate style
        if edge[0].startswith("topic_") or edge[1].startswith("topic_"):
            # Topic-paper connection (thicker, more visible)
            net.add_edge(edge[0], edge[1], width=2, color="#666666")
        else:
            # Paper-paper connection (thinner)
            net.add_edge(edge[0], edge[1], width=0.5, color="#cccccc")
    
    # Add custom HTML & JS to make the visualization more interactive
    custom_html = """
    <div style="position:absolute; top:10px; left:10px; z-index:999; background:white; padding:10px; border-radius:5px; box-shadow:0 0 10px rgba(0,0,0,0.2);">
        <h3>Paper Bridge Finder</h3>
        <p>🌉 Highlighted nodes are bridge papers between research areas</p>
        <p>Click on any node to open the paper</p>
    </div>
    """
    
    # Configure additional options
    net.set_options("""
    const options = {
        "interaction": {
            "hover": true,
            "navigationButtons": true,
            "keyboard": true
        },
        "physics": {
            "barnesHut": {
                "avoidOverlap": 0.2
            },
            "minVelocity": 0.75
        }
    }
    """)
    
    # Save interactive visualization
    output_path = VIS_DIR / output
    net.save_graph(str(output_path))
    
    # Add custom HTML to the generated file
    with open(output_path, 'r', encoding='utf-8') as f:
        html_content = f.read()
    
    # Insert custom HTML after the card-body div
    html_content = html_content.replace('<div id="mynetwork" class="card-body"></div>', 
                                       f'<div id="mynetwork" class="card-body"></div>{custom_html}')
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"Saved interactive graph visualization to {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Create visualizations of the paper knowledge graph")
    parser.add_argument("--papers", default="papers.json", help="Input papers JSON filename")
    parser.add_argument("--graph", default="paper_graph.gpickle", help="Input graph pickle filename")
    parser.add_argument("--output-static", default="graph_visualization.png", help="Output static visualization filename")
    parser.add_argument("--output-interactive", default="interactive_graph.html", help="Output interactive visualization filename")
    
    args = parser.parse_args()
    
    ensure_dirs()
    
    # Load data
    G = load_graph(args.graph)
    papers = load_papers(args.papers)
    topics = load_topics()
    
    # Create visualizations
    create_static_visualization(G, papers, topics, args.output_static)
    create_interactive_visualization(G, papers, topics, args.output_interactive)
    
    print("Visualizations created successfully!")

if __name__ == "__main__":
    main() 