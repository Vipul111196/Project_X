# RAPTOR Scripts Guide

This guide explains how to use the batch processing and interactive UI scripts for the RAPTOR system.

## 1. Batch QA Processor

The batch processor `batch_qa_processor.py` allows you to process multiple questions against a RAPTOR tree and save the results in JSON format.

### Features

- Process multiple question files at once
- Save detailed results including answers and retrieved context
- Group results by batch for easy analysis
- Support for various question formats (JSON, text files)

### Usage

```bash
python batch_qa_processor.py --questions ./path/to/questions --output ./results --batch-name experiment1
```

### Command Line Arguments

- `--config`: Path to config file (default: "config.yaml")
- `--tree`: Path to RAPTOR tree file (if different from config)
- `--questions`: Directory containing question files (required)
- `--pattern`: File pattern for question files (default: "*.json")
- `--output`: Output directory for results (default: "results")
- `--batch-name`: Optional name for this batch

### Question File Format

The script supports the following JSON formats:

1. Array of questions:
```json
[
  {"id": "q1", "question": "What is the meaning of X?"},
  {"id": "q2", "question": "How does system Y work?"}
]
```

2. Object with questions array:
```json
{
  "questions": [
    {"id": "q1", "question": "What is the meaning of X?"},
    {"id": "q2", "question": "How does system Y work?"}
  ]
}
```

3. Simple question objects:
```json
{"question": "What is the meaning of X?"}
```

### Output Format

The script creates a directory for each batch with individual result files and a combined `batch_results.json` file:

```
results/
└── batch_20250513_123456/
    ├── result_1.json
    ├── result_2.json
    └── batch_results.json
```

Each result includes:
- Original question
- Generated answer
- Retrieved context
- Metadata about retrieval (layer information, timestamps, etc.)

## 2. Interactive UI

The `raptor_ui.py` script provides a Streamlit-based user interface for interacting with RAPTOR trees.

### Features

- Selectable QA models (while maintaining the same embedding model as the tree)
- Interactive conversation history
- Display of both answers and retrieved context
- Reset button to start new conversations

### Requirements

```bash
pip install streamlit
```

### Usage

```bash
streamlit run raptor_ui.py
```

### Interface Components

- **Question Input**: Text area to enter your question
- **Answer Display**: Shows the generated answer
- **Context Viewer**: Expandable section showing the retrieved context
- **Model Selection**: Dropdown to choose different QA models
- **Reset Button**: Clears conversation history

### Configuring Available Models

The UI automatically detects available QA models in the codebase. To add custom models:

1. Create a new model class in `src/components/QAModels.py` that inherits from `BaseQAModel`
2. Implement the required `answer_question` method
3. Add the model to the `get_available_models()` function in `raptor_ui.py`

## 3. Integration Example

Here's how to use these scripts as part of a larger workflow:

```bash
# 1. Build the RAPTOR tree first
python tree_generator.py --config config.yaml

# 2. Process a set of benchmark questions
python batch_qa_processor.py --questions ./benchmark/questions --output ./benchmark/results

# 3. Launch the UI for interactive exploration
streamlit run raptor_ui.py
```

## 4. Programmatic Usage

You can also integrate these scripts into your own Python code:

```python
from src.components import RetrievalAugmentation, RetrievalAugmentationConfig
from src.models import get_qa_model

# Initialize configuration and models
config = load_config("config.yaml")
qa_model = get_qa_model(config)

# Set up RetrievalAugmentation
rac = RetrievalAugmentationConfig(qa_model=qa_model)
ra = RetrievalAugmentation(config=rac)

# Load a pre-built tree
with open("dataset/tree/raptor_model", "rb") as f:
    import pickle
    tree = pickle.load(f)
    ra.tree = tree
    ra.retriever = ra._init_retriever(tree)

# Process questions
answer = ra.answer_question("Your question here?")
print(answer)
```

## Troubleshooting

Common issues and solutions:

1. **Failed to load tree**: Ensure the tree file exists and is properly formatted.
   - Check file permissions
   - Verify the tree was saved with a compatible version of RAPTOR

2. **Model not loading**: Check API keys and model availability.
   - Set `OPENAI_API_KEY` environment variable for OpenAI models
   - Check HuggingFace token for HF models

3. **UI not launching**: Install required dependencies.
   - `pip install streamlit`
   - Ensure Python environment has required packages

4. **Large output in batch mode**: Consider limiting context size.
   - Adjust `max_tokens` parameter in `RetrievalAugmentation.retrieve()`
   - Filter or truncate contexts in post-processing
