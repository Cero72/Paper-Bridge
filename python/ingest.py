#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import time
import argparse
import requests
import feedparser
import asyncio
import httpx
from typing import List, Dict, Any, Optional
from pathlib import Path
import random
import logging
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
ARXIV_API_BASE = "http://export.arxiv.org/api/query"
S2_API_BASE = "https://api.semanticscholar.org/v1"
DATA_DIR = Path("data/raw")
REQUEST_RATE_LIMIT = 3  # Max 3 requests per second for arXiv API
MAX_RETRIES = 5
RETRY_MIN_WAIT = 1  # seconds
RETRY_MAX_WAIT = 30  # seconds

def ensure_data_dir():
    """Ensure the data directory exists."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

# Define custom exceptions for better retry control
class ArxivApiError(Exception):
    """Exception raised for arXiv API errors."""
    pass

class SemanticScholarApiError(Exception):
    """Exception raised for Semantic Scholar API errors."""
    pass

class TemporaryApiError(Exception):
    """Exception raised for temporary API errors that should be retried."""
    pass

def search_by_topic(topic: str, max_results: int = 50) -> List[str]:
    """
    Search for papers on arXiv by topic and return a list of arXiv IDs.
    
    Args:
        topic: Topic to search for
        max_results: Maximum number of results to return
        
    Returns:
        List of arXiv IDs for papers matching the topic
    """
    logger.info(f"Searching for papers on topic: {topic}")
    
    # Construct the search query
    search_query = f'all:"{topic}"'
    params = {
        "search_query": search_query,
        "max_results": max_results,
        "sortBy": "submittedDate",
        "sortOrder": "descending"
    }
    
    try:
        # Make API request
        response = requests.get(ARXIV_API_BASE, params=params)
        
        if response.status_code != 200:
            logger.error(f"arXiv API error: {response.status_code}")
            return []
        
        # Parse with feedparser
        feed = feedparser.parse(response.text)
        
        # Extract arXiv IDs
        arxiv_ids = []
        for entry in feed.entries:
            # Extract ID from the URL in the id field
            arxiv_id = entry.id.split("/abs/")[-1]
            if "v" in arxiv_id:
                arxiv_id = arxiv_id.split("v")[0]  # Remove version number
            arxiv_ids.append(arxiv_id)
            logger.info(f"Found paper: {arxiv_id} - {entry.title}")
        
        logger.info(f"Found {len(arxiv_ids)} papers on topic: {topic}")
        return arxiv_ids
        
    except Exception as e:
        logger.error(f"Error searching for papers on topic {topic}: {e}")
        return []

@retry(
    stop=stop_after_attempt(MAX_RETRIES),
    wait=wait_exponential(multiplier=1, min=RETRY_MIN_WAIT, max=RETRY_MAX_WAIT),
    retry=retry_if_exception_type((httpx.HTTPError, ConnectionError, TemporaryApiError)),
    reraise=True
)
async def fetch_arxiv_async(paper_id: str) -> Dict[str, Any]:
    """
    Fetch paper metadata from arXiv API asynchronously with retry logic.
    
    Args:
        paper_id: arXiv ID (e.g., '2303.08774')
        
    Returns:
        Dict containing paper metadata
    """
    # Clean the ID (remove version if present)
    clean_id = paper_id.split('v')[0] if 'v' in paper_id else paper_id
    
    # Make API request
    params = {"id_list": clean_id, "max_results": 1}
    
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(ARXIV_API_BASE, params=params)
            
            # Check status code and handle different types of errors
            if response.status_code == 200:
                # Success - continue processing
                pass
            elif response.status_code in (429, 503):  
                # Rate limit or service unavailable - should retry
                logger.warning(f"Rate limit or service unavailable from arXiv API: {response.status_code}")
                await asyncio.sleep(random.uniform(1, 5))  # Add jitter
                raise TemporaryApiError(f"Temporary arXiv API error: {response.status_code}")
            elif 500 <= response.status_code < 600:
                # Server error - may want to retry
                logger.warning(f"Server error from arXiv API: {response.status_code}")
                raise TemporaryApiError(f"arXiv API server error: {response.status_code}")
            else:
                # Client error or other error - don't retry
                logger.error(f"arXiv API error: {response.status_code}")
                raise ArxivApiError(f"arXiv API error: {response.status_code}")
            
            # Parse with feedparser
            feed = feedparser.parse(response.text)
            
            if not feed.entries:
                logger.warning(f"No entries found for {paper_id}")
                raise ArxivApiError(f"No entries found for {paper_id}")
            
            entry = feed.entries[0]
            
            # Extract metadata
            title = entry.title
            abstract = entry.summary
            published = entry.published  # Format: "2023-03-15T17:45:23Z"
            year = published.split("-")[0] if "-" in published else "Unknown Year"
            
            # Extract authors
            authors = [author.name for author in entry.get("authors", [])]
            
            # Create metadata dictionary
            metadata = {
                "id": clean_id,
                "title": title,
                "abstract": abstract,
                "year": year,
                "authors": authors,
                "source": "arxiv",
                "url": f"https://arxiv.org/abs/{clean_id}"
            }
            
            logger.info(f"Successfully fetched metadata for {paper_id}")
            return metadata
            
    except (httpx.HTTPError, ConnectionError) as e:
        logger.warning(f"Network error when fetching {paper_id}: {e}")
        raise  # Will be retried
    except Exception as e:
        logger.error(f"Unexpected error when fetching {paper_id}: {e}")
        raise  # Non-retryable exception

def fetch_arxiv(paper_id: str) -> Dict[str, Any]:
    """
    Synchronous wrapper for fetch_arxiv_async.
    
    Args:
        paper_id: arXiv ID (e.g., '2303.08774')
        
    Returns:
        Dict containing paper metadata
    """
    try:
        return asyncio.run(fetch_arxiv_async(paper_id))
    except Exception as e:
        logger.error(f"Failed to fetch paper {paper_id} after {MAX_RETRIES} retries: {e}")
        # Return minimal metadata to avoid breaking the pipeline
        return {
            "id": paper_id,
            "title": f"Failed to fetch paper {paper_id}",
            "abstract": "Metadata not available due to API error",
            "year": "Unknown",
            "authors": [],
            "source": "arxiv",
            "url": f"https://arxiv.org/abs/{paper_id}",
            "error": str(e)
        }

@retry(
    stop=stop_after_attempt(MAX_RETRIES),
    wait=wait_exponential(multiplier=1, min=RETRY_MIN_WAIT, max=RETRY_MAX_WAIT),
    retry=retry_if_exception_type((httpx.HTTPError, ConnectionError, TemporaryApiError)),
    reraise=True
)
async def fetch_semantic_scholar_async(paper_id: str, expand_references: bool = True, expand_citations: bool = True) -> Dict[str, Any]:
    """
    Fetch paper metadata and references/citations from Semantic Scholar API asynchronously with retry logic.
    
    Args:
        paper_id: arXiv ID (e.g., '2303.08774')
        expand_references: Whether to fetch paper references
        expand_citations: Whether to fetch papers that cite this one
        
    Returns:
        Dict containing paper metadata and references/citations
    """
    # Convert arXiv ID to S2 format
    s2_paper_id = f"ARXIV:{paper_id}"
    
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            # Make API request for paper details
            response = await client.get(f"{S2_API_BASE}/paper/{s2_paper_id}")
            
            # Handle different response statuses
            if response.status_code == 200:
                # Success - continue processing
                pass
            elif response.status_code == 404:
                # Paper not found - don't retry
                logger.warning(f"Paper {paper_id} not found in Semantic Scholar")
                return {}
            elif response.status_code in (429, 503):
                # Rate limit or service unavailable - should retry
                logger.warning(f"Rate limit or service unavailable from S2 API: {response.status_code}")
                await asyncio.sleep(random.uniform(1, 5))  # Add jitter
                raise TemporaryApiError(f"Temporary S2 API error: {response.status_code}")
            elif 500 <= response.status_code < 600:
                # Server error - may want to retry
                logger.warning(f"Server error from S2 API: {response.status_code}")
                raise TemporaryApiError(f"S2 API server error: {response.status_code}")
            else:
                # Client error or other error - don't retry
                logger.error(f"S2 API error: {response.status_code}")
                raise SemanticScholarApiError(f"S2 API error: {response.status_code}")
            
            paper_data = response.json()
            
            # Add references and citations if requested
            related_papers = {"references": [], "citations": []}
            
            if expand_references and "references" in paper_data:
                related_papers["references"] = paper_data["references"]
            
            if expand_citations:
                try:
                    # Make separate API call for citations
                    citations_response = await client.get(f"{S2_API_BASE}/paper/{s2_paper_id}/citations")
                    if citations_response.status_code == 200:
                        related_papers["citations"] = citations_response.json()
                except Exception as e:
                    logger.warning(f"Error fetching citations for {paper_id}: {e}")
            
            # Combine paper data with related papers
            paper_data["related"] = related_papers
            
            logger.info(f"Successfully fetched S2 data for {paper_id}")
            return paper_data
            
    except (httpx.HTTPError, ConnectionError) as e:
        logger.warning(f"Network error when fetching S2 data for {paper_id}: {e}")
        raise  # Will be retried
    except Exception as e:
        logger.error(f"Unexpected error when fetching S2 data for {paper_id}: {e}")
        raise  # Non-retryable exception

def fetch_semantic_scholar(paper_id: str, expand_references: bool = True, expand_citations: bool = True) -> Dict[str, Any]:
    """
    Synchronous wrapper for fetch_semantic_scholar_async.
    
    Args:
        paper_id: arXiv ID (e.g., '2303.08774')
        expand_references: Whether to fetch paper references
        expand_citations: Whether to fetch papers that cite this one
        
    Returns:
        Dict containing paper metadata and references/citations
    """
    try:
        return asyncio.run(fetch_semantic_scholar_async(paper_id, expand_references, expand_citations))
    except Exception as e:
        logger.error(f"Failed to fetch S2 data for {paper_id} after {MAX_RETRIES} retries: {e}")
        return {}

async def crawl_papers_async(seed_ids: List[str], max_papers: int = 800, hops: int = 2) -> Dict[str, Any]:
    """
    Crawl papers starting from seed IDs asynchronously.
    
    Args:
        seed_ids: List of arXiv IDs to start from
        max_papers: Maximum number of papers to crawl
        hops: Number of citation/reference hops to follow
        
    Returns:
        Dict mapping paper IDs to their metadata
    """
    papers = {}
    queue = [(id, 0) for id in seed_ids]  # (paper_id, hop_distance)
    visited = set()
    
    # Rate limiting semaphore (3 requests/second)
    semaphore = asyncio.Semaphore(REQUEST_RATE_LIMIT)
    
    async def fetch_with_rate_limit(paper_id, hop):
        async with semaphore:
            # Fetch from arXiv
            try:
                arxiv_data = await fetch_arxiv_async(paper_id)
                papers[paper_id] = arxiv_data
                print(f"Fetched {paper_id}: {arxiv_data['title']}")
                
                # Rate limiting
                await asyncio.sleep(1)  # Be nice to the arXiv API
                
                # If we haven't reached max hops, expand via Semantic Scholar
                if hop < hops:
                    s2_data = await fetch_semantic_scholar_async(paper_id)
                    
                    # Add references to queue
                    if "related" in s2_data and "references" in s2_data["related"]:
                        for ref in s2_data["related"]["references"]:
                            if "arxivId" in ref and ref["arxivId"]:
                                if ref["arxivId"] not in visited:
                                    queue.append((ref["arxivId"], hop + 1))
                    
                    # Add citations to queue
                    if "related" in s2_data and "citations" in s2_data["related"]:
                        for cit in s2_data["related"]["citations"]:
                            if "arxivId" in cit and cit["arxivId"]:
                                if cit["arxivId"] not in visited:
                                    queue.append((cit["arxivId"], hop + 1))
                
            except Exception as e:
                print(f"Error fetching {paper_id}: {e}")
    
    batch_size = 10  # Process in batches of 10 papers
    
    while queue and len(papers) < max_papers:
        # Take the next batch
        current_batch = []
        while queue and len(current_batch) < batch_size:
            paper_id, hop = queue.pop(0)
            if paper_id not in visited:
                visited.add(paper_id)
                current_batch.append((paper_id, hop))
        
        if not current_batch:
            break
        
        # Process the batch concurrently
        tasks = [fetch_with_rate_limit(paper_id, hop) for paper_id, hop in current_batch]
        await asyncio.gather(*tasks)
        
    return papers

def crawl_papers(seed_ids: List[str], max_papers: int = 800, hops: int = 2) -> Dict[str, Any]:
    """
    Synchronous wrapper for crawl_papers_async.
    
    Args:
        seed_ids: List of arXiv IDs to start from
        max_papers: Maximum number of papers to crawl
        hops: Number of citation/reference hops to follow
        
    Returns:
        Dict mapping paper IDs to their metadata
    """
    return asyncio.run(crawl_papers_async(seed_ids, max_papers, hops))

def save_papers(papers: Dict[str, Any], filename: str = "papers.json"):
    """Save papers to JSON file."""
    ensure_data_dir()
    filepath = DATA_DIR / filename
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(papers, f, indent=2, ensure_ascii=False)
    
    print(f"Saved {len(papers)} papers to {filepath}")

def main():
    parser = argparse.ArgumentParser(description="Fetch paper metadata from arXiv and expand via Semantic Scholar")
    parser.add_argument("--ids", nargs="+", help="arXiv IDs to fetch")
    parser.add_argument("--topic1", type=str, help="First topic to search for papers")
    parser.add_argument("--topic2", type=str, help="Second topic to search for papers")
    parser.add_argument("--num_papers", type=int, default=50, help="Number of papers to fetch per topic")
    parser.add_argument("--max", type=int, default=800, help="Maximum number of papers to fetch")
    parser.add_argument("--hops", type=int, default=2, help="Number of citation/reference hops to follow")
    parser.add_argument("--output", default="papers.json", help="Output JSON filename")
    
    args = parser.parse_args()
    
    # Check if we're searching by topic or by ID
    if args.topic1 or args.topic2:
        arxiv_ids = []
        if args.topic1:
            topic1_ids = search_by_topic(args.topic1, args.num_papers)
            arxiv_ids.extend(topic1_ids)
        
        if args.topic2:
            topic2_ids = search_by_topic(args.topic2, args.num_papers)
            arxiv_ids.extend(topic2_ids)
        
        if not arxiv_ids:
            logger.error("No papers found for the specified topics.")
            return
        
        papers = crawl_papers(arxiv_ids, max_papers=args.max, hops=args.hops)
        save_papers(papers, args.output)
    elif args.ids:
        papers = crawl_papers(args.ids, max_papers=args.max, hops=args.hops)
        save_papers(papers, args.output)
    else:
        logger.error("Either --ids or --topic1/--topic2 must be provided.")
        parser.print_help()

if __name__ == "__main__":
    main() 