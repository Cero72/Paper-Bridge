#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import sys
import asyncio
import subprocess
from typing import Dict, List, Any, Optional
from pathlib import Path
import networkx as nx
import pickle
from fastapi import FastAPI, HTTPException, Query, BackgroundTasks
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import uuid
import threading
import time

# Import our modules
from bridge_rank import load_graph, load_papers, find_bridge_papers, explain_bridge

# Constants
DATA_DIR = Path("data")
GRAPH_DIR = Path("data")
RESULTS_DIR = Path("data/results")
API_PORT = 8081

app = FastAPI(
    title="Paper Bridge API",
    description="API for finding papers that bridge between research topics",
    version="0.1.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize global variables
G = None
papers = None

JOBS_FILE = RESULTS_DIR / "jobs.json"

# Helper functions for job management

def load_jobs():
    if JOBS_FILE.exists():
        with open(JOBS_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}

def save_jobs(jobs):
    with open(JOBS_FILE, 'w', encoding='utf-8') as f:
        json.dump(jobs, f, indent=2)

def update_job(job_id, **kwargs):
    jobs = load_jobs()
    if job_id not in jobs:
        jobs[job_id] = {}
    jobs[job_id].update(kwargs)
    save_jobs(jobs)

def get_job(job_id):
    jobs = load_jobs()
    return jobs.get(job_id, None)

# Request/Response models
class BridgePaperRequest(BaseModel):
    topic_a: str
    topic_b: str
    top_k: int = 5
    alpha: float = 1.0
    beta: float = 0.3
    gamma: float = 0.1

class PaperInfo(BaseModel):
    paper_id: str
    title: str
    year: str
    authors: List[str]
    abstract: str
    score: float
    url: str
    explanation: Dict[str, Any]

class BridgePaperResponse(BaseModel):
    topic_a: str
    topic_b: str
    papers: List[PaperInfo]

class FeedbackRequest(BaseModel):
    paper_id: str
    liked: bool
    timestamp: str
    user_id: Optional[str] = None
    comments: Optional[str] = None

class IngestRequest(BaseModel):
    arxiv_ids: List[str]
    area_name: str
    max_papers: int = 20
    hops: int = 1

@app.on_event("startup")
async def startup_event():
    """Load graph and papers on startup."""
    global G, papers
    
    try:
        # Load graph
        graph_path = GRAPH_DIR / "paper_graph.gpickle"
        if graph_path.exists():
            # Use pickle to load the graph instead of read_gpickle
            with open(graph_path, 'rb') as f:
                G = pickle.load(f)
            print(f"Loaded graph with {len(G.nodes)} nodes and {len(G.edges)} edges")
        else:
            print("Warning: Graph file not found. API will not work until graph is created.")
            
        # Load papers
        papers_path = Path("data/raw") / "papers.json"
        if papers_path.exists():
            with open(papers_path, 'r', encoding='utf-8') as f:
                papers = json.load(f)
            print(f"Loaded {len(papers)} papers")
        else:
            print("Warning: Papers file not found. API will not work until papers are ingested.")
            
    except Exception as e:
        print(f"Error during startup: {e}")

@app.get("/")
async def root():
    """Root endpoint."""
    return {"message": "Paper Bridge API is running. Use /recommend to get paper recommendations."}

@app.get("/paper/{arxiv_id}")
async def get_paper(arxiv_id: str):
    """Get paper metadata by arXiv ID."""
    global papers
    
    # Check if papers are loaded
    if papers is None:
        papers_path = Path("data/raw") / "papers.json"
        if papers_path.exists():
            with open(papers_path, 'r', encoding='utf-8') as f:
                papers = json.load(f)
        else:
            print("Warning: Papers file not loaded. Will attempt to fetch directly from arXiv.")
            papers = {}
    
    # Check if paper exists in our database
    if arxiv_id in papers:
        return papers[arxiv_id]
    
    # If not found, try to fetch it directly from arXiv
    try:
        print(f"Paper {arxiv_id} not found in database. Fetching from arXiv...")
        
        # Import the ingest function to fetch from arXiv
        sys.path.append(str(Path(__file__).parent))
        from ingest import fetch_arxiv
        
        # Fetch paper directly
        paper_data = fetch_arxiv(arxiv_id)
        
        if paper_data:
            # Cache the result in memory (but don't save to file yet)
            papers[arxiv_id] = paper_data
            return paper_data
        else:
            raise HTTPException(status_code=404, detail=f"Paper {arxiv_id} not found on arXiv")
    except Exception as e:
        print(f"Error fetching paper {arxiv_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error fetching paper: {str(e)}")

@app.post("/ingest")
async def ingest_papers(request: IngestRequest, background_tasks: BackgroundTasks):
    if not request.arxiv_ids:
        raise HTTPException(status_code=400, detail="No arXiv IDs provided")
    
    # Create a unique job ID
    job_id = str(uuid.uuid4())
    job_file = RESULTS_DIR / f"job_{job_id}.json"
    now = time.time()
    update_job(job_id,
        status="pending",
        created=now,
        updated=now,
        area_name=request.area_name,
        arxiv_ids=request.arxiv_ids,
        max_papers=request.max_papers,
        hops=request.hops,
        result_file=str(job_file)
    )
    
    # Start the pipeline in the background
    def run_job():
        try:
            update_job(job_id, status="running", updated=time.time())
            ids_str = " ".join(request.arxiv_ids)
            ingest_cmd = f"python python/ingest.py --ids {ids_str} --max {request.max_papers} --hops {request.hops}"
            subprocess.run(ingest_cmd, shell=True, check=True)
            pipeline_steps = [
                "python python/embed.py",
                "python python/topic_model_simple.py --n-clusters 5",
                "python python/graph.py"
            ]
            for cmd in pipeline_steps:
                subprocess.run(cmd, shell=True, check=True)
            # After pipeline, find bridge papers for all topic pairs (or just save the processed data)
            # For now, just save a marker file
            with open(job_file, 'w', encoding='utf-8') as f:
                json.dump({"message": "Pipeline complete for job", "job_id": job_id}, f)
            update_job(job_id, status="complete", updated=time.time())
        except Exception as e:
            update_job(job_id, status="failed", error=str(e), updated=time.time())
    
    # Use threading to avoid blocking event loop
    threading.Thread(target=run_job, daemon=True).start()
    
    return {"job_id": job_id}

@app.get("/job_status/{job_id}")
async def job_status(job_id: str):
    job = get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return {"job_id": job_id, "status": job.get("status", "unknown"), "updated": job.get("updated"), "created": job.get("created"), "error": job.get("error")}

@app.get("/job_result/{job_id}")
async def job_result(job_id: str):
    job = get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job.get("status") != "complete":
        return {"status": job.get("status"), "message": "Job not complete yet"}
    result_file = job.get("result_file")
    if not result_file or not os.path.exists(result_file):
        raise HTTPException(status_code=404, detail="Result file not found")
    with open(result_file, 'r', encoding='utf-8') as f:
        return json.load(f)

@app.get("/topics")
async def get_topics():
    """Get all available topics."""
    try:
        # Load topic info
        topic_info_path = DATA_DIR / "topic_info.csv"
        if not topic_info_path.exists():
            raise HTTPException(status_code=404, detail="Topic info file not found")
        
        import pandas as pd
        df = pd.read_csv(topic_info_path)
        
        # Convert to list of topics
        topics = []
        for _, row in df.iterrows():
            if row["Topic"] != -1:  # Skip outlier topic
                topics.append({
                    "id": f"topic_{row['Topic']}",
                    "name": row["Name"],
                    "count": row["Count"]
                })
        
        return {"topics": topics}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading topics: {e}")

@app.post("/recommend", response_model=BridgePaperResponse)
async def recommend(request: BridgePaperRequest):
    """Get paper recommendations that bridge between two topics."""
    global G, papers
    
    # Check if graph and papers are loaded
    if G is None or papers is None:
        raise HTTPException(status_code=503, detail="Service not ready. Graph or papers not loaded.")
    
    # Format topic IDs if integer was provided
    topic_a = request.topic_a if request.topic_a.startswith('topic_') else f"topic_{request.topic_a}"
    topic_b = request.topic_b if request.topic_b.startswith('topic_') else f"topic_{request.topic_b}"
    
    # Check if topics exist
    if topic_a not in G or topic_b not in G:
        raise HTTPException(status_code=404, detail=f"One or both topics not found: {topic_a}, {topic_b}")
    
    # Find bridge papers
    bridge_papers = find_bridge_papers(topic_a, topic_b, G, papers, 
                                      request.top_k, request.alpha, request.beta, request.gamma)
    
    # Generate explanations and format response
    result_papers = []
    for paper_id, score in bridge_papers:
        # Get paper details
        if paper_id in papers:
            paper = papers[paper_id]
            explanation = explain_bridge(paper_id, topic_a, topic_b, G, papers)
            
            # Create paper info
            paper_info = PaperInfo(
                paper_id=paper_id,
                title=paper.get("title", "Unknown"),
                year=paper.get("year", "Unknown"),
                authors=paper.get("authors", []),
                abstract=paper.get("abstract", ""),
                score=score,
                url=paper.get("url", f"https://arxiv.org/abs/{paper_id}"),
                explanation=explanation
            )
            
            result_papers.append(paper_info)
    
    # Return response
    return BridgePaperResponse(
        topic_a=topic_a,
        topic_b=topic_b,
        papers=result_papers
    )

@app.post("/feedback")
async def feedback(request: FeedbackRequest):
    """Store user feedback on recommendations."""
    # In a real implementation, store feedback in a database
    # For the prototype, just log it and save to a file
    
    print(f"Received feedback for paper {request.paper_id}: {'👍' if request.liked else '👎'}")
    
    # Save feedback to file
    try:
        # Ensure directory exists
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        
        feedback_file = RESULTS_DIR / "feedback.jsonl"
        
        # Append to file (one JSON object per line)
        with open(feedback_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(request.dict()) + '\n')
            
        return {"status": "success", "message": "Feedback recorded"}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error recording feedback: {e}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=API_PORT) 