# RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval

## Overview

RAPTOR is a hierarchical chunking system for RAG (Retrieval-Augmented Generation) pipelines. It creates a tree structure of documents with summaries at different levels, allowing for more efficient and context-aware retrieval.

## Key Components

![RAPTOR Architecture](raptor.jpg)

### 1. Document Processing Pipeline

The document processing pipeline converts raw documents into a hierarchical tree structure:

```
Document → Chunks → Summaries → Tree Structure
```

### 2. Tree Structure

The tree consists of:

- **Leaf nodes**: Original text chunks from the document
- **Internal nodes**: Summaries of their child nodes
- **Root node**: High-level summary of the entire document

### 3. Retrieval Process

Retrieval leverages the tree structure for more efficient and context-aware search:

1. **Query Processing**: Convert user query to embedding
2. **Tree Traversal**: Navigate from root to most relevant branches
3. **Chunk Selection**: Identify most relevant leaf nodes
4. **Context Assembly**: Combine selected chunks with their ancestry context

## Using RAPTOR with QASPER Dataset

### Prerequisites

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Process QASPER dataset:

```bash
python QASPER_dataset.py --config config.yaml
```

### Model Configuration

The `config.yaml` file allows you to choose between OpenAI and Hugging Face models:

```yaml
# In config.yaml
raptor:
  models:
    # Choose provider: "openai" or "huggingface"
    provider: "huggingface"
  
  # Choose embedding model provider
  embedding:
    provider: "sentence_transformers"  # "sentence_transformers", "openai", or "huggingface"
```

#### Using OpenAI Models

To use OpenAI models:

1. Set the environment variable for your API key:
   ```
   set OPENAI_API_KEY=your_api_key_here
   ```
   
2. Configure the model parameters in `config.yaml`:
   ```yaml
   openai:
     qa_model: "gpt-4o-mini"
     summarization_model: "gpt-4o-mini"
     qa_params:
       max_tokens: 256
       temperature: 0.7
   ```

3. To use OpenAI for embeddings as well:
   ```yaml
   embedding:
     provider: "openai"
     model_name: "text-embedding-3-small"
     dimensions: 1536  # Optional
   ```

   Available embedding models:
   - `text-embedding-3-small` (Fastest, most cost-effective)
   - `text-embedding-3-large` (Higher quality)
   - `text-embedding-ada-002` (Legacy model)

#### Using Hugging Face Models

To use Hugging Face models:

1. Set the environment variable for your token if needed:
   ```
   set HF_TOKEN=your_token_here
   ```
   
2. Configure the model parameters in `config.yaml`:
   ```yaml
   huggingface:
     unified_model: true
     model_name: "meta-llama/Llama-3.1-8B-Instruct"
     device: "auto"
     quantization: "fp16"
   ```

### Building the RAPTOR Tree

1. Configure tree building parameters in `config.yaml`
2. Run the tree generator:

```bash
run_tree_generator.bat
```

Or directly with:

```bash
python tree_generator.py --config config.yaml
```

### Running RAG Pipeline with RAPTOR

```bash
run_rag_pipeline.bat
```

## Workflow Steps

1. **Data Preparation**:
   - Download and process QASPER dataset using `QASPER_dataset.py`
   - Dataset will be organized into JSON files and plain text files

2. **Tree Generation**:
   - Documents are split into chunks
   - Chunks are grouped and summarized recursively
   - Tree structure is stored with nodes at each level

3. **Embedding Creation**:
   - Each node in the tree is embedded using a language model
   - Embeddings are stored for efficient retrieval

4. **Retrieval Process**:
   - Query is embedded using same model
   - Tree is traversed to find relevant chunks
   - Relevant chunks and their context are retrieved

5. **Response Generation**:
   - Retrieved chunks are passed to LLM as context
   - LLM generates answer based on retrieved information

## Advantages of RAPTOR

1. **Preserves document structure**: Maintains hierarchical relationships between content
2. **Multi-level context**: Provides both detailed chunks and summarized context
3. **Efficient retrieval**: Reduces search space through tree traversal
4. **Better reasoning**: Provides both specific details and higher-level context

## Configuration

Customize parameters in `config.yaml`:

- Chunk size and overlap
- Summary generation settings
- Embedding models
- Tree traversal strategies

## Additional Resources

For more information on the RAPTOR architecture and implementation details, refer to the comments in the code files and the paper documentation.
