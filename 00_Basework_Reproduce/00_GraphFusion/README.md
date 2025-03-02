# GraphFusion

GraphFusion is a sophisticated knowledge graph construction pipeline designed to extract, process, and fuse domain-specific knowledge from textual data. It leverages Natural Language Processing (NLP) techniques and Large Language Models (LLMs) to identify concepts, extract relationships, and build a coherent knowledge graph.

## Overview

The GraphFusion pipeline consists of four main steps:

1. **Entity Extraction**: Identifies key concepts from textual abstracts using BERTopic and other NLP techniques
2. **Triplet Extraction**: Extracts relationships between concepts using LLMs to create knowledge graph triplets
3. **Graph Fusion**: Merges and reconciles extracted triplets to create a cohesive knowledge graph
4. **Neo4j Integration**: Stores the knowledge graph in Neo4j for visualization and querying (in development)

## Features

- Automated concept extraction from academic abstracts or other textual data
- LLM-powered relationship extraction with predefined semantic relation types
- Knowledge graph fusion to create a comprehensive and consistent graph
- Support for Neo4j integration to visualize and query the knowledge graph
- Configurable pipeline with YAML-based configuration
- Detailed logging and progress tracking

## Project Structure

```
GraphFusion/
├── config.yaml                  # Main configuration file
├── main.py                      # Entry point for the pipeline
├── requirements.txt             # Python dependencies
├── experiments/                 # Jupyter notebooks for experimentation
├── inputs/                      # Input data files
│   ├── abstracts.txt            # Text corpus for processing
│   └── relation_types.json      # Definitions of relation types
├── logs/                        # Logging output
├── outputs/                     # Pipeline output files
│   ├── experiments/             # Output from experiment notebooks
│   └── pipeline/                # Output from main pipeline runs
├── prompts/                     # LLM prompt templates
│   ├── prompt_fusion.txt        # Template for graph fusion
│   └── prompt_tpextraction.txt  # Template for triplet extraction
└── src/                         # Source code
    ├── logger.py                # Logging configuration
    ├── exception.py             # Exception handling
    └── components/              # Pipeline components
        ├── entity_extraction.py     # Concept extraction module
        ├── triplets_extraction.py   # Relation extraction module
        ├── graph_fusion.py          # Graph fusion module
        ├── graph_to_neo4j.py        # Neo4j integration module
        └── models.py                # LLM model definitions
```

## Installation

1. Clone the repository:
```bash
cd GraphFusion
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

3. Configure your environment variables:
Create a `.env` file in the root directory with:
```
OPENAI_API_KEY=your_openai_api_key
NEO4J_URI=your_neo4j_uri
NEO4J_USERNAME=your_neo4j_username
NEO4J_PASSWORD=your_neo4j_password
```

## Usage

### Configuration

Edit the `config.yaml` file to customize the pipeline:

```yaml
step 1:  # Entity Extraction
  name: "entity_extraction"
  data_path: "inputs/abstracts.txt"
  sample_data: "yes"                # Use sample of data (yes/no)
  sample_size: 200                  # Sample size if using samples
  # ... other configuration options

step 2:  # Triplet Extraction
  name: "triples_extraction"
  # ... configuration options

step 3:  # Graph Fusion
  name: "triples_fusion"
  # ... configuration options

step 4:  # Neo4j Integration
  name: "saving_to_neo4j"
  # ... configuration options
```

### Running the Pipeline

To run the complete pipeline:

```bash
python main.py
```

This will execute each step of the pipeline in sequence, from entity extraction to neo4j integration.

### Step-by-Step Usage

You can also run individual steps of the pipeline by modifying the `main.py` file to only execute specific functions.

### Notebooks

The `experiments/` directory contains Jupyter notebooks that demonstrate each step of the pipeline:

- `01_test_topic_extraction.ipynb`: Demonstrates entity extraction
- `02_test_tpextraction.ipynb`: Demonstrates triplet extraction
- `03_Graph_fusion.ipynb`: Demonstrates graph fusion
- `04_neo4j.ipynb`: Demonstrates Neo4j integration

## Predefined Relation Types

The system uses the following predefined relation types:

1. **Compare**: Represents a relationship where a comparison is made between entities
2. **Part-of**: Denotes a constituent or component relationship
3. **Conjunction**: Indicates a logical or semantic connection between entities
4. **Evaluate-for**: Represents an evaluative relationship
5. **Is-a-Prerequisite-of**: Implies one entity is a characteristic or prerequisite for another
6. **Used-for**: Denotes a functional relationship
7. **Hyponym-Of**: Establishes a hierarchical relationship between entities

## Neo4j Integration

GraphFusion supports integration with Neo4j for graph visualization and querying. To use this feature:

1. Set up a Neo4j database
2. Configure the Neo4j connection details in the `.env` file
3. Run the pipeline with Neo4j integration enabled

## Contributing

Contributions to GraphFusion are welcome! Please feel free to submit a Pull Request.

## License

[MIT License](LICENSE)

## Acknowledgments

- This project uses LangChain for LLM integration
- BERTopic for topic modeling
- Neo4j for graph database storage
- Sentence Transformers for embeddings
