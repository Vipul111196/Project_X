# RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval

RAPTOR is a hierarchical chunking system for RAG (Retrieval-Augmented Generation) pipelines that creates a tree structure of documents with summaries at different levels, allowing for more efficient and context-aware retrieval.

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Process the QASPER dataset (if not already done)
python QASPER_dataset.py --config config.yaml

# 3. Build the RAPTOR tree
python tree_generator.py --config config.yaml

# 4. Run RAG pipeline with RAPTOR
python pipeline_for_QA.py --config config.yaml

# 5. Process multiple questions in batch mode
python batch_qa_processor.py --questions ./dataset/qas --output ./results

# 6. Launch the interactive UI
streamlit run raptor_ui.py
```

## Repository Structure

```
00_raptor/
├── config.yaml                # Main configuration file
├── tree_generator.py          # Builds the hierarchical tree structure
├── tree_retrieval.py          # Tool for exploring the tree structure
├── pipeline_for_QA.py         # RAG pipeline using RAPTOR
├── batch_qa_processor.py      # Process multiple questions in batch mode
├── raptor_ui.py               # Interactive Streamlit UI for Q&A
├── raptor_demo.py             # Simple demonstration script
├── QASPER_dataset.py          # Dataset preparation script
├── RAPTOR_USAGE.md            # Detailed documentation
└── src/                       # Core RAPTOR components
    ├── components/            # Implementation of tree building and retrieval
    │   ├── cluster_tree_builder.py # Hierarchical clustering implementation
    │   ├── tree_retriever.py  # Recursive tree traversal for retrieval
    │   └── ...                # Other component files
    └── models.py              # Model integration utilities
```

## Configuration

All settings are managed through the `config.yaml` file. Key configuration sections:

- **Dataset**: QASPER dataset settings and subset selection
- **RAPTOR Pipeline**: Tree building parameters and model settings
- **Models**: Choose between OpenAI and Hugging Face LLMs
- **Embedding**: Configure embedding model providers
- **Logging**: Logging settings

Example configuration:

```yaml
raptor:
  models:
    provider: "openai"        # Choose "openai" or "huggingface"
  embedding:
    provider: "openai"        # "openai", "sentence_transformers", or "huggingface"
    model_name: "text-embedding-3-small"
```

## Using RAPTOR

### 1. Building the Tree

Configure parameters in `config.yaml` and run:

```bash
python tree_generator.py --config config.yaml
```

This builds a hierarchical tree structure of documents with:
- Leaf nodes: Original text chunks
- Internal nodes: Summaries of their child nodes
- Root node: High-level summary

### 2. Exploring the Tree Structure

The `tree_retrieval.py` script lets you explore the tree and understand how documents are organized:

```bash
# List all layers in the tree
python tree_retrieval.py "dataset/tree/raptor_model" --list

# View nodes at a specific layer
python tree_retrieval.py "dataset/tree/raptor_model" --list --layer 2

# Get the full subtree for a specific node
python tree_retrieval.py "dataset/tree/raptor_model" --node 125 --output "./reports/subtree_node_125.json"
```

### 3. Running the RAG Pipeline

```bash
python pipeline_for_QA.py --config config.yaml
```

### 4. Processing Questions in Batch Mode

To process multiple questions at once, use the `batch_qa_processor.py` script. This is useful for evaluating the model on a set of questions or for generating answers for a FAQ-style dataset.

```bash
python batch_qa_processor.py --questions ./dataset/qas --output ./results
```

### 5. Using the Interactive UI

RAPTOR includes an interactive UI for a more user-friendly question-answering experience. To launch the UI, run:

```bash
streamlit run raptor_ui.py
```

## Cluster Hierarchy Reports

RAPTOR now generates detailed reports showing the clustering hierarchy:

- **JSON Report**: Structured data about all nodes, layers, and their relationships
- **Text Report**: Human-readable format showing which nodes were clustered together

Reports are automatically generated when building the tree and saved to the configured report directory.

## Integration with Other Algorithms

RAPTOR can be used alongside other retrieval algorithms for comparison or ensemble approaches:

1. **Standalone Use**: Run RAPTOR independently using the scripts in this folder
2. **Integration**: Import RAPTOR components into other Python modules:

```python
from src.components import RetrievalAugmentation, RetrievalAugmentationConfig
from src.models import get_embedding_model, get_summarization_model

# Initialize RAPTOR with your models
config = RetrievalAugmentationConfig(...)
raptor = RetrievalAugmentation(config)

# Add your documents
raptor.add_documents(documents)

# Use RAPTOR for retrieval
results = raptor.retrieve(query)
```

## Using Different LLM Providers

### OpenAI

1. Set your API key:
   ```bash
   set OPENAI_API_KEY=your_api_key_here
   ```

2. Update `config.yaml`:
   ```yaml
   raptor:
     models:
       provider: "openai"
       openai:
         qa_model: "gpt-4o-mini"
         summarization_model: "gpt-4o-mini"
   ```

### Hugging Face

1. Set your token (if needed):
   ```bash
   set HF_TOKEN=your_token_here
   ```

2. Update `config.yaml`:
   ```yaml
   raptor:
     models:
       provider: "huggingface"
       huggingface:
         model_name: "meta-llama/Llama-3.1-8B-Instruct"
         device: "auto"
         quantization: "fp16"
   ```

## Advantages of RAPTOR

- **Preserves document structure**: Maintains hierarchical relationships
- **Multi-level context**: Provides both detailed chunks and summarized context
- **Efficient retrieval**: Reduces search space through tree traversal
- **Better reasoning**: Provides both specific details and higher-level context

## Additional Resources

For more detailed information, refer to `RAPTOR_USAGE.md`, `SCRIPT_GUIDE.md`, or explore the implementation in the `src/components` directory.
