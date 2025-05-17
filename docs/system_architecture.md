# Paper Bridge Finder - System Architecture

## Overview

The Paper Bridge Finder is a system designed to help researchers discover papers that connect different research areas. It uses natural language processing, embedding techniques, and graph algorithms to identify "bridge papers" - publications that serve as conceptual bridges between distinct research domains.

## Core Components

The system is composed of these main components:

### 1. Data Ingestion (python/ingest.py)

This module is responsible for:
- Fetching paper metadata from ArXiv API
- Extracting additional information from Semantic Scholar API
- Building a comprehensive paper dataset
- Handling error recovery and retry logic for API failures
- Saving papers to a structured JSON format

**Key Features:**
- Asynchronous and concurrent API requests for better performance
- Rate limiting to respect API usage constraints
- Robust error handling with exponential backoff retry logic
- Crawling papers by following citations and references

### 2. Embedding Generation (python/embed.py)

This module creates vector representations (embeddings) of papers using:
- Title and abstract text
- SentenceTransformer models for high-quality semantic embeddings

**Key Features:**
- Batched processing for efficiency
- Configurable embedding models
- Persistent storage of embeddings for reuse

### 3. Topic Modeling (python/topic_model_simple.py)

This component:
- Clusters papers into topics using K-means
- Creates topic labels and descriptions
- Visualizes the topic landscape

**Key Features:**
- Configurable number of clusters (topics)
- Topic visualization using dimensionality reduction
- Topic naming based on representative terms

### 4. Knowledge Graph Construction (python/graph.py)

This module builds a knowledge graph where:
- Nodes represent papers and topics
- Edges represent relationships between them
- Edge weights capture the strength of connections

**Key Features:**
- NetworkX-based graph representation
- Weighted connections between papers and topics
- Serialization for persistence

### 5. Bridge Paper Detection (python/bridge_rank.py)

The core algorithm for:
- Finding papers that connect different research topics
- Ranking papers by their "bridge strength"
- Explaining why certain papers act as bridges

**Key Features:**
- Multiple ranking metrics (centrality, relevance, novelty)
- Configurable scoring parameters
- Detailed explanations for bridge paper recommendations

### 6. Graph Visualization (python/visualize_graph.py)

Creates visualizations of the knowledge graph:
- Static PNG visualizations
- Interactive HTML visualizations using PyVis
- Topic-paper relationship diagrams

**Key Features:**
- Customizable visualization settings
- Interactive node exploration
- Visual highlighting of bridge papers

### 7. API Server (python/api.py)

RESTful API that provides:
- Paper metadata retrieval
- Bridge paper recommendations
- Topic information
- Background processing of ingestion requests

**Key Features:**
- FastAPI-based implementation
- Asynchronous request handling
- Background task processing
- CORS support for web frontend

### 8. Web User Interface (ui/streamlit_app.py)

A Streamlit-based UI that offers:
- Input fields for ArXiv IDs from different research areas
- Paper preview and metadata display
- Interactive graph visualization
- Bridge paper recommendations with explanations

**Key Features:**
- Real-time status updates
- Expandable paper details
- Interactive controls for graph exploration
- User feedback mechanisms

## Data Flow

1. **Paper Ingestion**:
   - User enters ArXiv IDs for two research areas
   - System fetches metadata from ArXiv and Semantic Scholar
   - Papers are saved to JSON format in data/raw/

2. **Processing Pipeline**:
   - Papers are embedded using SentenceTransformer
   - Embeddings are clustered into topics
   - A knowledge graph is constructed
   - Bridge papers are identified

3. **Visualization and Results**:
   - The knowledge graph is visualized
   - Bridge papers are ranked and displayed
   - Explanations are generated
   - User can provide feedback

## Folder Structure

```
paper-bridge-finder/
├── data/                  # Data files
│   ├── raw/               # Raw paper data
│   ├── embeddings/        # Paper embeddings
│   └── results/           # Generated results
├── docs/                  # Documentation
├── python/                # Python modules
│   ├── ingest.py          # Paper ingestion
│   ├── embed.py           # Embedding generation
│   ├── topic_model_simple.py  # Topic modeling
│   ├── graph.py           # Graph construction
│   ├── bridge_rank.py     # Bridge paper detection
│   ├── visualize_graph.py # Graph visualization
│   ├── api.py             # API server
│   └── test_suite.py      # Testing suite
├── ui/                    # User interface
│   └── streamlit_app.py   # Streamlit app
└── requirements.txt       # Project dependencies
```

## Deployment

The Paper Bridge Finder has two main components to deploy:

1. **API Server**:
   ```
   cd paper-bridge-finder
   uvicorn python.api:app --host 0.0.0.0 --port 8081
   ```

2. **Streamlit UI**:
   ```
   cd paper-bridge-finder
   streamlit run ui/streamlit_app.py
   ```

## Configuration

The system behavior can be configured through various parameters:

1. **API Configuration** (python/api.py):
   - `API_PORT`: Port for the API server (default: 8081)
   - `DATA_DIR`: Directory for data files
   - `GRAPH_DIR`: Directory for graph files

2. **Ingestion Configuration** (python/ingest.py):
   - `REQUEST_RATE_LIMIT`: Rate limit for API requests
   - `MAX_RETRIES`: Maximum retry attempts for API calls
   - `RETRY_MIN_WAIT`/`RETRY_MAX_WAIT`: Retry wait times

3. **Topic Modeling** (python/topic_model_simple.py):
   - `n_clusters`: Number of topics to create

4. **Bridge Ranking** (python/bridge_rank.py):
   - `alpha`: Weight for relevance score
   - `beta`: Weight for centrality score
   - `gamma`: Weight for novelty score

## Performance Considerations

- **Memory Usage**: Large paper sets can consume significant memory during embedding
- **API Rate Limits**: Respect the rate limits of ArXiv and Semantic Scholar APIs
- **Processing Time**: Topic modeling and graph construction can be computationally intensive

## Error Handling

The system includes:
- Retry logic for transient API failures
- Graceful degradation when papers can't be fetched
- Logging for all operations
- Exception handling throughout the processing pipeline

## Future Improvements

- User authentication system
- Persistent caching for API requests
- More sophisticated topic modeling algorithms
- Enhanced visualization options
- Improved explanation generation for bridge papers 