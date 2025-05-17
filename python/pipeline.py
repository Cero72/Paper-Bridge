#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import argparse
import subprocess
import time
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def run_command(cmd, description):
    """Run a command and log its output."""
    logger.info(f"Starting {description}...")
    start_time = time.time()
    
    try:
        result = subprocess.run(
            cmd, 
            check=True,
            text=True,
            capture_output=True
        )
        logger.info(f"{description} completed successfully in {time.time() - start_time:.2f} seconds")
        logger.info(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"{description} failed: {e}")
        logger.error(f"Error output: {e.stderr}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Run the Paper Bridge Finder pipeline")
    
    # Topic-based search arguments
    topic_group = parser.add_argument_group('Topic-based search')
    topic_group.add_argument("--topic1", type=str, help="First topic to search for papers")
    topic_group.add_argument("--topic2", type=str, help="Second topic to search for papers")
    topic_group.add_argument("--num_papers", type=int, default=15, help="Number of papers to fetch per topic")
    
    # ID-based search arguments
    id_group = parser.add_argument_group('ID-based search')
    id_group.add_argument("--ids", nargs="+", help="List of arXiv IDs to ingest")
    
    # Common parameters
    parser.add_argument("--n-clusters", type=int, default=2, help="Number of topic clusters to create")
    parser.add_argument("--skip-ingest", action="store_true", help="Skip the ingestion step")
    parser.add_argument("--skip-embed", action="store_true", help="Skip the embedding step")
    parser.add_argument("--skip-topic", action="store_true", help="Skip the topic modeling step")
    parser.add_argument("--skip-graph", action="store_true", help="Skip the graph building step")
    parser.add_argument("--skip-vis", action="store_true", help="Skip the visualization step")
    
    # Bridge finding parameters
    parser.add_argument("--find-bridges", action="store_true", help="Find bridge papers between topics")
    parser.add_argument("--top-k", type=int, default=5, help="Number of top bridge papers to return")
    
    # API server
    parser.add_argument("--start-api", action="store_true", help="Start the API server after pipeline completion")
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.ids and not (args.topic1 or args.topic2):
        parser.error("Either --ids or at least one of --topic1/--topic2 must be provided")
    
    # Step 1: Data Ingestion
    if not args.skip_ingest:
        ingest_cmd = ["python", "python/ingest.py"]
        
        if args.topic1 or args.topic2:
            # Topic-based search
            if args.topic1:
                ingest_cmd.extend(["--topic1", args.topic1])
            if args.topic2:
                ingest_cmd.extend(["--topic2", args.topic2])
            if args.num_papers:
                ingest_cmd.extend(["--num_papers", str(args.num_papers)])
        else:
            # ID-based search
            ingest_cmd.extend(["--ids"] + args.ids)
        
        # Always use 0 hops for better performance
        ingest_cmd.extend(["--hops", "0"])
        
        success = run_command(
            ingest_cmd,
            "Data Ingestion"
        )
        if not success:
            logger.error("Pipeline failed at ingestion step. Exiting.")
            return False
    
    # Step 2: Embedding Generation
    if not args.skip_embed:
        success = run_command(
            ["python", "python/embed.py"],
            "Embedding Generation"
        )
        if not success:
            logger.error("Pipeline failed at embedding step. Exiting.")
            return False
    
    # Step 3: Topic Modeling
    if not args.skip_topic:
        success = run_command(
            ["python", "python/topic_model.py", "--n-clusters", str(args.n_clusters)],
            "Topic Modeling"
        )
        if not success:
            logger.error("Pipeline failed at topic modeling step. Exiting.")
            return False
    
    # Step 4: Knowledge Graph Construction
    if not args.skip_graph:
        success = run_command(
            ["python", "python/graph.py"],
            "Knowledge Graph Construction"
        )
        if not success:
            logger.error("Pipeline failed at graph construction step. Exiting.")
            return False
    
    # Step 5: Graph Visualization
    if not args.skip_vis:
        success = run_command(
            ["python", "python/visualize_graph.py"],
            "Graph Visualization"
        )
        if not success:
            logger.error("Pipeline failed at visualization step. Exiting.")
            return False
    
    # Step 6: Find Bridge Papers
    if args.find_bridges and args.n_clusters >= 2:
        logger.info("Finding bridge papers between topics...")
        
        # When clustering into 2 topics, we know they are topic_0 and topic_1
        if args.n_clusters == 2:
            bridge_cmd = ["python", "python/bridge_rank.py", "--topic-a", "0", "--topic-b", "1"]
            if args.top_k:
                bridge_cmd.extend(["--top-k", str(args.top_k)])
            
            success = run_command(
                bridge_cmd,
                "Bridge Finding"
            )
            if not success:
                logger.error("Pipeline failed at bridge finding step. Exiting.")
                return False
        else:
            # For more than 2 clusters, find all pairs
            topic_pairs = []
            for i in range(args.n_clusters):
                for j in range(i+1, args.n_clusters):
                    topic_pairs.append((i, j))
            
            logger.info(f"Finding bridges between {len(topic_pairs)} topic pairs...")
            
            for topic_a, topic_b in topic_pairs:
                bridge_cmd = [
                    "python", "python/bridge_rank.py", 
                    "--topic-a", str(topic_a), 
                    "--topic-b", str(topic_b),
                    "--output", f"bridge_results_{topic_a}_{topic_b}.json"
                ]
                if args.top_k:
                    bridge_cmd.extend(["--top-k", str(args.top_k)])
                
                success = run_command(
                    bridge_cmd,
                    f"Bridge Finding between topic_{topic_a} and topic_{topic_b}"
                )
                if not success:
                    logger.warning(f"Failed to find bridges between topic_{topic_a} and topic_{topic_b}")
    
    # Step 7: Start API Server (if requested)
    if args.start_api:
        logger.info("Starting API server...")
        try:
            # Use subprocess.Popen to start the server in the background
            process = subprocess.Popen(
                ["python", "python/api.py"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            # Wait a moment for the server to start
            time.sleep(3)
            
            # Check if the process is still running
            if process.poll() is None:
                logger.info("API server started successfully. Access at http://localhost:8081")
                logger.info("Press Ctrl+C to stop the server when done.")
                
                # Keep the script running until interrupted
                try:
                    while True:
                        time.sleep(1)
                except KeyboardInterrupt:
                    logger.info("Stopping API server...")
                    process.terminate()
                    process.wait()
                    logger.info("API server stopped.")
            else:
                stdout, stderr = process.communicate()
                logger.error(f"API server failed to start: {stderr}")
                return False
        except Exception as e:
            logger.error(f"Error starting API server: {e}")
            return False
    
    logger.info("Pipeline completed successfully!")
    return True

if __name__ == "__main__":
    sys.exit(0 if main() else 1) 