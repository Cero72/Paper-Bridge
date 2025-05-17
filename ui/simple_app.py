#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import subprocess
import time
import streamlit as st
import pandas as pd
import webbrowser
from pathlib import Path
import matplotlib.pyplot as plt
import json
import datetime

# Constants
DATA_DIR = Path("data")
VIS_DIR = DATA_DIR / "visualizations"

# Wrap all top-level Streamlit calls in a main function check to avoid duplicate elements
if __name__ == "__main__":
    # Set page config
    st.set_page_config(
        page_title="Paper Bridge Finder",
        page_icon="🌉",
        layout="wide"
    )
    
    # Initialize session state for cache invalidation
    if 'search_timestamp' not in st.session_state:
        st.session_state.search_timestamp = datetime.datetime.now().isoformat()
    if 'current_search' not in st.session_state:
        st.session_state.current_search = ""
    
    # Custom CSS
    st.markdown("""
    <style>
        /* General styling */
        .main-title {
            font-size: 3rem !important;
            color: #1E88E5;
            text-align: center;
            margin-bottom: 2rem;
        }
        .header-line {
            height: 3px;
            background-color: #1E88E5;
            margin-bottom: 2rem;
        }
        
        /* Card styling */
        .result-card {
            background-color: #f8f9fa;
            border-radius: 12px;
            padding: 20px;
            margin-bottom: 20px;
            border-left: 5px solid #1E88E5;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
            transition: transform 0.2s, box-shadow 0.2s;
        }
        .result-card:hover {
            transform: translateY(-3px);
            box-shadow: 0 6px 12px rgba(0, 0, 0, 0.1);
        }
        
        /* Typography */
        .bridge-title {
            color: #1565C0;
            font-weight: bold;
            margin-top: 0;
            margin-bottom: 15px;
        }
        .paper-score {
            color: #E53935;
            font-weight: bold;
            background-color: rgba(229, 57, 53, 0.1);
            padding: 3px 8px;
            border-radius: 4px;
        }
        
        /* Info and context boxes */
        .info-box {
            background-color: #E3F2FD;
            border-radius: 8px;
            padding: 20px;
            margin-bottom: 20px;
            border-left: 5px solid #2196F3;
        }
        
        /* Button styling - overrides for consistency */
        .stButton > button {
            font-weight: 500;
            border-radius: 6px;
            transition: all 0.2s;
        }
        .stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        }
        
        /* Tab styling */
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
        }
        .stTabs [data-baseweb="tab"] {
            height: 50px;
            border-radius: 6px 6px 0 0;
            padding: 0 20px;
            font-weight: 500;
        }
        
        /* Link styling */
        a {
            color: #1E88E5;
            text-decoration: none;
            font-weight: 500;
        }
        a:hover {
            text-decoration: underline;
        }
        
        /* Responsive adjustments */
        @media (max-width: 768px) {
            .main-title {
                font-size: 2rem !important;
            }
            .result-card {
                padding: 15px;
            }
        }
    </style>
    """, unsafe_allow_html=True)

def run_pipeline(topic1, topic2, num_papers=10, top_k=5, n_clusters=2):
    """Run the Paper Bridge Finder pipeline with the specified topics."""
    with st.spinner(f"Finding bridge papers between '{topic1}' and '{topic2}'..."):
        try:
            # Update cache key to invalidate previous results
            st.session_state.search_timestamp = datetime.datetime.now().isoformat()
            st.session_state.current_search = f"{topic1}_{topic2}_{num_papers}_{n_clusters}_{top_k}"
            
            # Clear any existing results before running new search
            try:
                # Remove all possible bridge results files based on cluster number
                # Default bridge results file
                bridge_files = [DATA_DIR / "results" / "bridge_results.json"]
                
                # Add pair-specific files for the number of clusters
                for i in range(n_clusters):
                    for j in range(i+1, n_clusters):
                        bridge_files.append(DATA_DIR / "results" / f"bridge_results_{i}_{j}.json")
                
                # Remove each file if it exists (without showing info messages)
                for bridge_file in bridge_files:
                    if os.path.exists(bridge_file):
                        os.remove(bridge_file)
                
                # Remove visualization files
                vis_files = [
                    VIS_DIR / "graph_visualization.png",
                    VIS_DIR / "interactive_graph.html"
                ]
                for file in vis_files:
                    if os.path.exists(file):
                        os.remove(file)
            except Exception as e:
                st.error(f"Error clearing old results: {e}")
            
            # Build the command to run the pipeline
            cmd = [
                "python", "python/pipeline.py",
                "--topic1", topic1,
                "--topic2", topic2,
                "--num_papers", str(num_papers),
                "--n-clusters", str(n_clusters),
                "--find-bridges",
                "--top-k", str(top_k)
            ]
            
            # Show the command being run in a collapsible section
            with st.expander("Command Details", expanded=False):
                st.code(' '.join(cmd), language="bash")
            
            # Run the command
            process = subprocess.run(
                cmd,
                text=True,
                capture_output=True
            )
            
            # Check if the process was successful
            if process.returncode == 0:
                st.success(f"Successfully found bridge papers between '{topic1}' and '{topic2}'!")
                
                # Check if any bridge results file was created
                bridge_file_found = False
                
                # Try the default bridge results file first
                default_bridge_file = DATA_DIR / "results" / "bridge_results.json"
                if os.path.exists(default_bridge_file):
                    bridge_file_found = True
                    with st.expander("Technical Details", expanded=False):
                        st.info(f"Bridge results saved to {default_bridge_file}")
                # If not found and n_clusters > 2, check for pair-specific files
                elif n_clusters > 2:
                    for i in range(n_clusters):
                        for j in range(i+1, n_clusters):
                            pair_file = DATA_DIR / "results" / f"bridge_results_{i}_{j}.json"
                            if os.path.exists(pair_file):
                                bridge_file_found = True
                                with st.expander("Technical Details", expanded=False):
                                    st.info(f"Bridge results saved to {pair_file}")
                
                if not bridge_file_found:
                    st.error("Pipeline ran successfully but did not create any bridge results files.")
                    with st.expander("Pipeline Output", expanded=True):
                        st.code(process.stdout, language="text")
                    return False
                    
                # Display any output from the pipeline for information
                if process.stdout:
                    with st.expander("Pipeline Output", expanded=False):
                        st.code(process.stdout, language="text")
                
                return True
            else:
                st.error(f"Error running pipeline")
                st.error(process.stderr)
                return False
                
        except Exception as e:
            st.error(f"Error: {str(e)}")
            import traceback
            st.error(traceback.format_exc())
            return False

def load_bridge_results():
    """Load bridge paper results from JSON file."""
    try:
        # Use cache key to ensure fresh results
        cache_key = st.session_state.search_timestamp
        
        # Get the number of clusters from the current search
        n_clusters = 2  # Default
        if 'current_search' in st.session_state and st.session_state.current_search:
            search_parts = st.session_state.current_search.split('_')
            if len(search_parts) >= 4:  # topic1_topic2_numpapers_nclusters_topk
                try:
                    n_clusters = int(search_parts[3])
                except ValueError:
                    pass
        
        # Check different possible result files based on number of clusters
        results_path = None
        
        # Try default file first
        default_path = DATA_DIR / "results" / "bridge_results.json"
        if default_path.exists():
            results_path = default_path
        # If n_clusters > 2, try pair-specific files
        else:
            # First check for any bridge results files
            bridge_files = list(Path(DATA_DIR / "results").glob("bridge_results_*.json"))
            if bridge_files:
                # Take the first one we find
                results_path = bridge_files[0]
            # If no bridge files found but we know the cluster count, check specific pairs
            elif n_clusters > 2:
                # Check all possible topic pairs
                for i in range(n_clusters):
                    for j in range(i+1, n_clusters):
                        pair_path = DATA_DIR / "results" / f"bridge_results_{i}_{j}.json"
                        if pair_path.exists():
                            # Take the first one we find
                            results_path = pair_path
                            break
                    if results_path:
                        break
        
        if not results_path:
            st.warning("No bridge results found. Please run the pipeline first.")
            with st.expander("Debug Information", expanded=False):
                st.write(f"Searched for results with {n_clusters} clusters")
                st.write(f"Tried: {default_path}")
                st.write(f"Also tried: glob pattern bridge_results_*.json")
                if n_clusters > 2:
                    for i in range(n_clusters):
                        for j in range(i+1, n_clusters):
                            st.write(f"Also tried: {DATA_DIR / 'results' / f'bridge_results_{i}_{j}.json'}")
            return None
        
        # Create a debug expander for detailed information
        with st.expander("Debug Information", expanded=False):
            st.write(f"Loading results from: {results_path}")
            file_mtime = os.path.getmtime(results_path)
            file_time = datetime.datetime.fromtimestamp(file_mtime).isoformat()
            st.write(f"File modified: {file_time}")
            st.write(f"Current timestamp: {cache_key}")
            
            # Load and show data structure
            with open(results_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            st.write(f"Data keys: {list(data.keys())}")
            if "results" in data:
                st.write(f"Number of results: {len(data['results'])}")
            elif "bridge_papers" in data:
                st.write(f"Number of bridge papers: {len(data['bridge_papers'])}")
        
        with open(results_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        # Convert the results to the expected format
        # The bridge_results.json contains results in "results" field, not "papers"
        if "results" in data:
            # Read the raw papers file to get full paper details
            papers_path = DATA_DIR / "raw" / "papers.json"
            papers_data = {}
            if papers_path.exists():
                with open(papers_path, 'r', encoding='utf-8') as f:
                    papers_data = json.load(f)
            
            # Create the expected format
            formatted_results = {
                "topic_a": data.get("topic_a", "unknown"),
                "topic_b": data.get("topic_b", "unknown"),
                "papers": [],
                "timestamp": file_time,  # Use file timestamp instead of cache key
                "source_file": str(results_path)
            }
            
            # Convert each result to the expected paper format
            for result in data["results"]:
                paper_id = result.get("paper_id", "")
                paper_info = {
                    "id": paper_id,
                    "title": result.get("explanation", {}).get("title", "Unknown Title"),
                    "authors": [],  # Will fill from papers_data if available
                    "year": result.get("explanation", {}).get("year", "Unknown Year"),
                    "score": result.get("score", 0),
                    "url": f"https://arxiv.org/abs/{paper_id}"
                }
                
                # Get additional info from papers_data if available
                if paper_id in papers_data:
                    paper = papers_data[paper_id]
                    paper_info["authors"] = paper.get("authors", [])
                    if "url" in paper:
                        paper_info["url"] = paper["url"]
                
                formatted_results["papers"].append(paper_info)
            
            return formatted_results
        elif "bridge_papers" in data:
            # Handle the case where the results are in bridge_papers format
            formatted_results = {
                "topic_a": data.get("topic1", "unknown"),
                "topic_b": data.get("topic2", "unknown"),
                "papers": [],
                "timestamp": file_time,
                "source_file": str(results_path)
            }
            
            for paper in data.get("bridge_papers", []):
                paper_info = {
                    "id": paper.get("id", ""),
                    "title": paper.get("title", "Unknown Title"),
                    "authors": paper.get("authors", []),
                    "year": paper.get("year", "Unknown Year"),
                    "score": paper.get("score", 0),
                    "url": paper.get("url", f"https://arxiv.org/abs/{paper.get('id', '')}")
                }
                formatted_results["papers"].append(paper_info)
            
            return formatted_results
        else:
            # It might be directly formatted as we expect
            with st.expander("Debug Information", expanded=False):
                st.warning("Unexpected data format in bridge_results.json")
                st.json(data)
            return data
    except Exception as e:
        with st.expander("Error Details", expanded=False):
            st.error(f"Error loading bridge results: {str(e)}")
            import traceback
            st.code(traceback.format_exc(), language="python")
        return None

def display_topic_info():
    """Display topic information."""
    try:
        # Use cache key to ensure fresh results
        cache_key = st.session_state.search_timestamp
        
        topic_info_path = DATA_DIR / "topic_info.csv"
        if not topic_info_path.exists():
            return
        
        # Check file modification time
        file_mtime = os.path.getmtime(topic_info_path)
        
        df = pd.read_csv(topic_info_path)
        
        # Display the topics
        st.subheader("Research Areas")
        for _, row in df.iterrows():
            topic_id = row['Topic']
            topic_name = row['Name']
            paper_count = row['Count']
            st.markdown(f"**Topic {topic_id}:** {topic_name} ({paper_count} papers)")
    except Exception as e:
        st.error(f"Error displaying topic info: {e}")

def open_interactive_viz():
    """Open the interactive visualization directly without a link."""
    try:
        # Use cache key to ensure fresh results
        cache_key = st.session_state.search_timestamp
        
        viz_path = VIS_DIR / "interactive_graph.html"
        if not viz_path.exists():
            st.warning("No visualization available. Please run the pipeline first.")
            return False
        
        # Check file modification time
        file_mtime = os.path.getmtime(viz_path)
        
        # Instead of returning a URL, we'll embed the HTML directly
        with open(viz_path, 'r', encoding='utf-8') as f:
            html_content = f.read()
        
        st.components.v1.html(html_content, height=600, scrolling=True)
        return True
    except Exception as e:
        st.error(f"Error opening visualization: {e}")
        return False

def display_bridge_papers(results):
    """Display bridge papers with their scores."""
    if not results or "papers" not in results:
        st.warning("No bridge papers found in results.")
        return
    
    papers = results["papers"]
    if not papers:
        st.warning("No bridge papers found between these research areas.")
        return
    
    # Display search info and source file
    if 'current_search' in st.session_state and st.session_state.current_search:
        search_parts = st.session_state.current_search.split('_')
        if len(search_parts) >= 2:
            st.success(f"Successfully found bridge papers connecting **{search_parts[0]}** and **{search_parts[1]}**")
    
    # Show source file information if available
    if "source_file" in results:
        source_file = results["source_file"]
        source_file_name = os.path.basename(source_file)
        with st.expander("Source Details", expanded=False):
            st.info(f"Results loaded from: {source_file_name}")
            if "_" in source_file_name and source_file_name.startswith("bridge_results_"):
                parts = source_file_name.replace(".json", "").split("_")
                if len(parts) >= 3:
                    cluster1, cluster2 = parts[-2], parts[-1]
                    st.write(f"Shows connections between Topic {cluster1} and Topic {cluster2}")
    
    # Header for bridge papers section
    st.markdown("""
    <div style="text-align: center; margin-bottom: 20px;">
        <h2 style="color: #1E88E5;">🌉 Top Bridge Papers</h2>
        <p>These papers effectively connect the two research areas based on content, citations, and topic proximity.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Create a uniform grid for the papers
    for i, paper in enumerate(papers):
        with st.container():
            title = paper.get('title', 'Unknown Title')
            authors = paper.get('authors', [])
            year = paper.get('year', 'Unknown Year')
            score = paper.get('score', 0)
            url = paper.get('url', '#')
            
            # Safely create author string
            author_str = ', '.join(authors[:3]) if authors else 'Unknown Authors'
            if len(authors) > 3:
                author_str += '...'
                
            st.markdown(f"""
            <div class="result-card">
                <h3 class="bridge-title">#{i+1}: {title}</h3>
                <p><strong>Authors:</strong> {author_str}</p>
                <p><strong>Year:</strong> {year}</p>
                <p><strong>Bridge Score:</strong> <span class="paper-score">{score:.4f}</span></p>
                <p><a href="{url}" target="_blank">View Paper</a></p>
            </div>
            """, unsafe_allow_html=True)
    
    # Add explanation about bridge score
    with st.expander("About Bridge Scores", expanded=False):
        st.markdown("""
        Bridge scores measure how effectively a paper connects the two research domains:
        - **Higher scores** indicate papers that bridge concepts across domains better
        - Scores consider factors like paper citations, centrality, recency, and domain relevance
        - Papers are ranked based on their role in the network of connections between topics
        """)

def display_static_visualization():
    """Display the static graph visualization."""
    try:
        # Use cache key to ensure fresh results
        cache_key = st.session_state.search_timestamp
        
        viz_path = VIS_DIR / "graph_visualization.png"
        if viz_path.exists():
            # Check file modification time
            file_mtime = os.path.getmtime(viz_path)
            
            st.subheader("Static Graph Visualization")
            # Set a fixed width to make the visualization smaller
            st.image(str(viz_path), width=800)
        else:
            st.warning("No visualization available. Please run the pipeline first.")
    except Exception as e:
        st.error(f"Error displaying visualization: {e}")

def copy_viz_to_temp():
    """Copy visualization to a temporary location for easier access."""
    try:
        import shutil
        src_path = VIS_DIR / "interactive_graph.html"
        if not src_path.exists():
            return None
            
        # Create a user-friendly path in the temp directory
        import tempfile
        temp_dir = Path(tempfile.gettempdir()) / "paper_bridge_finder"
        temp_dir.mkdir(exist_ok=True)
        
        # Use timestamp in filename to prevent caching
        timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        dest_path = temp_dir / f"interactive_graph_{timestamp}.html"
        shutil.copy2(src_path, dest_path)
        
        return dest_path
    except Exception as e:
        st.error(f"Error copying visualization: {e}")
        return None

def display_clusters_info():
    """Display information about clusters and their optimal number."""
    st.markdown("""
    <div class="info-box">
        <h3>How Many Clusters Should You Use?</h3>
        <p>The number of clusters determines how papers are grouped into research areas:</p>
        <ul>
            <li><strong>2 clusters:</strong> Best for clearly defined research areas and finding direct bridges between them. This is the recommended setting for most use cases.</li>
            <li><strong>3-4 clusters:</strong> Useful when your research areas have distinct sub-areas or when exploring finer-grained connections.</li>
            <li><strong>5+ clusters:</strong> For complex or diverse research domains where you want to discover nuanced relationships.</li>
        </ul>
        <p>For optimal bridge paper discovery, we recommend using 2 clusters when comparing two distinct research areas.</p>
    </div>
    """, unsafe_allow_html=True)

def clear_cache():
    """Clear the cache and force a refresh."""
    st.session_state.search_timestamp = datetime.datetime.now().isoformat()
    st.session_state.current_search = ""
    # Clear any files to force regeneration
    try:
        # Remove all possible bridge results files based on cluster number
        # Default bridge results file
        bridge_files = [DATA_DIR / "results" / "bridge_results.json"]
        
        # Add pair-specific files for potential clusters (up to 5)
        for i in range(5):  # Support up to 5 clusters
            for j in range(i+1, 5):
                bridge_files.append(DATA_DIR / "results" / f"bridge_results_{i}_{j}.json")
        
        # Remove each file if it exists (without showing info messages)
        files_removed = []
        for bridge_file in bridge_files:
            if os.path.exists(bridge_file):
                os.remove(bridge_file)
                files_removed.append(str(bridge_file))
        
        # Remove visualization files
        vis_files = [
            VIS_DIR / "graph_visualization.png",
            VIS_DIR / "interactive_graph.html"
        ]
        for file in vis_files:
            if os.path.exists(file):
                os.remove(file)
                files_removed.append(str(file))
                
        # Only show a summary message if files were removed
        if files_removed:
            with st.expander("Cache Cleared", expanded=False):
                st.write(f"Removed {len(files_removed)} cached files")
                st.write(", ".join(files_removed))
                
    except Exception as e:
        st.error(f"Error clearing cache files: {e}")
        
    st.rerun()  # Updated from experimental_rerun

def show_results(topic1, topic2):
    """Show results after pipeline runs."""
    # Check if the results file exists
    results_path = DATA_DIR / "results" / "bridge_results.json"
    
    # If default path doesn't exist, check for cluster-specific files
    if not os.path.exists(results_path):
        # Look for any bridge results files
        bridge_files = list(Path(DATA_DIR / "results").glob("bridge_results_*.json"))
        
        if bridge_files:
            st.info(f"Found {len(bridge_files)} cluster-specific bridge files instead of the default bridge_results.json")
            
            # Use the first bridge results file found
            results_path = bridge_files[0]
            
            # Show all available bridge files for user information
            with st.expander("Available Bridge Files"):
                for bf in bridge_files:
                    st.write(f"- {bf}")
        else:
            st.error(f"Results file not found at {results_path}")
            return
    
    # Copy visualization to temp directory for easier access
    temp_viz_path = copy_viz_to_temp()
    
    # Show results
    results = load_bridge_results()
    if not results:
        st.error("Failed to load bridge results. Please check the logs and try again.")
        # Try to display raw file content
        try:
            with open(results_path, 'r', encoding='utf-8') as f:
                raw_content = f.read()
                with st.expander("Raw Results File Content"):
                    st.code(raw_content, language="json")
        except Exception as e:
            st.error(f"Error reading raw results file: {e}")
        return
        
    # If we have results but no papers
    if "papers" not in results or not results["papers"]:
        st.warning("No bridge papers were found between these research areas.")
        return
        
    # Display tabs for different views
    tab1, tab2, tab3, tab4 = st.tabs(["Bridge Papers", "Static Graph", "Interactive Graph", "Topic Information"])
    
    with tab1:
        display_bridge_papers(results)
    
    with tab2:
        # Display static visualization if it exists
        viz_path = VIS_DIR / "graph_visualization.png"
        if os.path.exists(viz_path):
            display_static_visualization()
        else:
            st.warning("Static graph visualization not found.")
    
    with tab3:
        # Direct HTML embedding of interactive visualization if it exists
        viz_path = VIS_DIR / "interactive_graph.html"
        if os.path.exists(viz_path):
            st.subheader("Interactive Graph (Zoom Out)")
            st.markdown("You can click on nodes to open papers directly in your browser.")
            open_interactive_viz()
            
            # Also provide the file path for manual opening
            if temp_viz_path:
                st.markdown(f"""
                **Alternative:** Open this file in your browser manually:
                ```
                {temp_viz_path}
                ```
                """)
        else:
            st.warning("Interactive graph visualization not found.")
    
    with tab4:
        # Display topic info if it exists
        topic_info_path = DATA_DIR / "topic_info.csv"
        if os.path.exists(topic_info_path):
            display_topic_info()
        else:
            st.warning("Topic information not found.")

def main():
    """Main function to run the Streamlit app."""
    # Header
    st.markdown('<h1 class="main-title">Paper Bridge Finder</h1>', unsafe_allow_html=True)
    st.markdown('<div class="header-line"></div>', unsafe_allow_html=True)
    
    st.markdown("""
    Discover papers that connect different research areas. Enter two research topics and 
    let the system find papers that effectively bridge between these domains.
    """)
    
    # Input section
    with st.container():
        col1, col2 = st.columns(2)
        
        with col1:
            topic1 = st.text_input("Research Area 1", "", key=f"topic1_{st.session_state.search_timestamp}")
        
        with col2:
            topic2 = st.text_input("Research Area 2", "", key=f"topic2_{st.session_state.search_timestamp}")
    
    # Parameters section
    with st.expander("Advanced Parameters", expanded=True):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            num_papers = st.slider("Number of papers per topic", min_value=5, max_value=50, value=15, step=5, 
                                  help="How many papers to fetch for each research area",
                                  key=f"num_papers_{st.session_state.search_timestamp}")
        
        with col2:
            n_clusters = st.slider("Number of clusters", min_value=2, max_value=5, value=2, step=1,
                                 help="How many topic clusters to create from the papers",
                                 key=f"n_clusters_{st.session_state.search_timestamp}")
        
        with col3:
            top_k = st.slider("Number of bridge papers to find", min_value=3, max_value=20, value=5, step=1,
                            help="How many top bridge papers to return",
                            key=f"top_k_{st.session_state.search_timestamp}")
        
        # Display information about optimal cluster number
        display_clusters_info()
    
    # Action buttons in a more consistent layout
    button_cols = st.columns([3, 1])
    
    # Primary search button
    with button_cols[0]:
        if st.button("Find Bridge Papers 🔍", type="primary", key=f"run_btn_{st.session_state.search_timestamp}", 
                    help="Run search with current parameters", use_container_width=True):
            if not topic1 or not topic2:
                st.error("Please enter both research areas.")
            else:
                success = run_pipeline(topic1, topic2, num_papers, top_k, n_clusters)
                if success:
                    show_results(topic1, topic2)
    
    # Refresh button
    with button_cols[1]:
        if st.button("🔄 Refresh", key=f"refresh_btn_{st.session_state.search_timestamp}", 
                   help="Clear the cache and refresh the app", use_container_width=True):
            clear_cache()
    
    # Check if results exist on page load (for when the app refreshes)
    if os.path.exists(DATA_DIR / "results" / "bridge_results.json"):
        # Only load if we don't have active search results displayed
        if not st.session_state.get('current_search'):
            results = load_bridge_results()
            if results:
                tab1, tab2, tab3, tab4 = st.tabs(["Bridge Papers", "Static Graph", "Interactive Graph", "Topic Information"])
                
                with tab1:
                    display_bridge_papers(results)
                
                with tab2:
                    display_static_visualization()
                
                with tab3:
                    st.subheader("Interactive Graph")
                    st.markdown("You can click on nodes to open papers directly in your browser.")
                    open_interactive_viz()
                
                with tab4:
                    display_topic_info()

if __name__ == "__main__":
    main() 