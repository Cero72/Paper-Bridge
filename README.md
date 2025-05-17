# Paper Bridge Finder

A system that discovers papers that connect different research areas by analyzing their semantic content and relationships.

## Overview

The Paper Bridge Finder helps researchers discover interdisciplinary papers that effectively bridge between different research domains. It works by:

1. **Retrieving papers** from arXiv based on topic keywords
2. **Creating embeddings** of paper abstracts using sentence transformers
3. **Clustering papers** into topics using unsupervised learning
4. **Building a knowledge graph** connecting papers and topics
5. **Identifying bridge papers** that effectively connect different research areas
6. **Visualizing the connections** between topics and papers

## Installation

### Prerequisites

- Python 3.8+
- Git (for cloning the repository)

### Installation Steps

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/paper-bridge-finder.git
   cd paper-bridge-finder
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Running the Application

### Streamlit Web Interface (Recommended)

The easiest way to use Paper Bridge Finder is through the Streamlit web interface:

1. Run the Streamlit app:
   ```bash
   streamlit run ui/simple_app.py
   ```
   Or on Windows, you can use the provided batch file:
   ```bash
   run_app.bat
   ```

2. Open your browser to the URL shown in the terminal (typically http://localhost:8501)

3. Enter two research areas you want to connect, adjust parameters if needed, and click "Find Bridge Papers 🔍"

### Command Line Usage

You can also run the complete pipeline from the command line:

```bash
python python/pipeline.py --topic1 "first_research_area" --topic2 "second_research_area" --num_papers 15 --find-bridges
```

### Example Queries

Try these example research area pairs:
- "quantum computing" and "machine learning"
- "economics" and "artificial intelligence"
- "neuroscience" and "reinforcement learning"
- "materials science" and "deep learning"

## Command Line Arguments

The pipeline supports the following arguments:

### Topic-based search
- `--topic1`: First topic to search for papers
- `--topic2`: Second topic to search for papers
- `--num_papers`: Number of papers to fetch per topic (default: 15)

### ID-based search
- `--ids`: List of specific arXiv IDs to analyze

### Other parameters
- `--n-clusters`: Number of topic clusters to create (default: 2)
- `--find-bridges`: Enable bridge paper detection
- `--top-k`: Number of top bridge papers to return (default: 5)
- `--skip-ingest`: Skip the data ingestion step
- `--skip-embed`: Skip the embedding generation step
- `--skip-topic`: Skip the topic modeling step
- `--skip-graph`: Skip the graph construction step
- `--skip-vis`: Skip the visualization step

## Visualizations

After running the pipeline, you can view:
- Static graph visualization: `data/visualizations/graph_visualization.png`
- Interactive graph: `data/visualizations/interactive_graph.html`
  - The interactive graph allows clicking on papers to open them directly
  - Bridge papers are highlighted with an orange-red border and 🌉 emoji

## Project Structure

```
.
├── data/                    # Data directory
│   ├── raw/                 # Raw paper data
│   ├── visualizations/      # Graph visualizations
│   └── results/             # Bridge paper results
├── python/                  # Python code
│   ├── pipeline.py          # Main pipeline script
│   ├── ingest.py            # Data ingestion from arXiv
│   ├── embed.py             # Embedding generation
│   ├── topic_model.py       # Topic modeling
│   ├── graph.py             # Knowledge graph construction
│   ├── visualize_graph.py   # Graph visualization
│   ├── bridge_rank.py       # Bridge paper identification
│   └── api.py               # API server
├── ui/                      # User interface
│   └── simple_app.py        # Streamlit application
├── run_app.bat              # Batch file to run the Streamlit app (Windows)
└── README.md                # This readme file
```

## Tips for Best Results

- **Number of Papers**: Increasing the number of papers (using `--num_papers`) improves the quality of the analysis but increases processing time. 15-20 papers per topic works well.

- **Number of Clusters**: For most searches, 2 clusters works best to clearly separate the research areas. Use more clusters (3-5) for discovering nuanced connections between sub-domains.

- **Bridge Papers**: The recommended number of bridge papers (using `--top-k`) is 5-10 for optimal visualization and analysis.

## Git Setup

If you're setting up this project in your own Git repository:

1. Initialize the repository:
   ```bash
   git init
   ```

2. Add the files:
   ```bash
   git add .
   ```

3. Create an initial commit:
   ```bash
   git commit -m "Initial commit of Paper Bridge Finder"
   ```

4. Link to your remote repository (replace the URL with your own):
   ```bash
   git remote add origin https://github.com/your-username/paper-bridge-finder.git
   ```

5. Push to the repository:
   ```bash
   git push -u origin main
   ```

## License

[MIT License](LICENSE) 