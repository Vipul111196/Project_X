#!/usr/bin/env python3
"""
RAPTOR Interactive UI

A Streamlit-based UI for interacting with RAPTOR trees.
This allows users to ask questions against a pre-built RAPTOR tree
and see both the answers and retrieved context.
"""

import os
import sys
import json
import yaml
import pickle
import logging
import streamlit as st
from typing import Dict, Any, List, Optional, Tuple

# Add parent directory to path if needed
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

# Import RAPTOR components
from src.components import RetrievalAugmentation, RetrievalAugmentationConfig
from src.components.QAModels import (BaseQAModel, GPT4ominiQAModel, 
                                    GPT4oQAModel, UnifiedQAModel)
from src.components.EmbeddingModels import (BaseEmbeddingModel, OpenAIEmbeddingModel, SBertEmbeddingModel)
from src.models import get_qa_model, get_summarization_model, get_embedding_model

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/streamlit_ui.log", mode="w")
    ]
)

# Constants
DEFAULT_CONFIG_PATH = "config.yaml"
CONFIG_SECTION = "raptor"

# Create logs directory if it doesn't exist
os.makedirs("logs", exist_ok=True)

def load_config(config_path: str = DEFAULT_CONFIG_PATH) -> Dict[str, Any]:
    """
    Load configuration from YAML file
    
    Args:
        config_path: Path to config file
        
    Returns:
        Configuration dictionary
    """
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        logging.info(f"Configuration loaded from {config_path}")
        return config
    except Exception as e:
        logging.error(f"Error loading configuration: {e}")
        st.error(f"Error loading configuration: {e}")
        return {}

def load_tree(tree_path: str) -> Any:
    """
    Load a previously serialized RAPTOR Tree object
    
    Args:
        tree_path: Path to pickled tree file
        
    Returns:
        Path to the tree file (for RetrievalAugmentation to load)
    """
    try:
        # Just verify the file exists
        with open(tree_path, 'rb') as f:
            pass
        logging.info(f"Tree file exists at {tree_path}")
        return tree_path  # Return the path instead of loading the tree
    except Exception as e:
        logging.error(f"Failed to access tree file at {tree_path}: {e}")
        st.error(f"Failed to access RAPTOR tree file: {e}")
        return None

def get_available_models() -> Tuple[Dict[str, type], Dict[str, type]]:
    """
    Get available QA and summarization model classes
    
    Returns:
        Tuple of (qa_models, summarization_models) dictionaries
    """
    qa_models = {
        "GPT-4o-mini": GPT4ominiQAModel,
        "GPT-4o": GPT4oQAModel,
        "UnifiedQA": UnifiedQAModel,
    }
    
    return qa_models

def initialize_retrieval_augmentation(
    tree_path, 
    qa_model_name: str, 
    embedding_model: Optional[OpenAIEmbeddingModel] = None
) -> RetrievalAugmentation:
    """
    Initialize the RetrievalAugmentation with specified models
    
    Args:
        tree_path: Path to the pickled RAPTOR tree file
        qa_model_name: Name of QA model to use
        embedding_model: Embedding model (optional)
        
    Returns:
        Initialized RetrievalAugmentation instance
    """
    qa_models = get_available_models()
    
    # Initialize QA model
    if qa_model_name in qa_models:
        qa_model = qa_models[qa_model_name]()
    else:
        qa_model = GPT4ominiQAModel()  # Default
    
    # Create config
    config = RetrievalAugmentationConfig(qa_model=qa_model)
    
    # Initialize RetrievalAugmentation with the tree path
    # The RetrievalAugmentation class will handle loading the tree from the path
    ra = RetrievalAugmentation(config=config, tree=tree_path)
    
    return ra

def render_answer(answer: str, context: str):
    """Render the answer and context in the UI"""
    st.markdown("### Answer:")
    st.write(answer)
    
    with st.expander("View Retrieved Context", expanded=False):
        st.markdown("### Context:")
        st.write(context)

def display_interface(tree_path: str):
    """
    Display the main Streamlit interface
    
    Args:
        tree_path: Path to the RAPTOR tree file
    """
    st.title("RAPTOR Interactive Q&A")
    
    # Load models
    qa_models = get_available_models()
    qa_model_names = list(qa_models.keys())
    
    # Sidebar
    st.sidebar.header("Settings")
    selected_qa_model = st.sidebar.selectbox(
        "Select QA Model:",
        qa_model_names,
        index=1  # Default to GPT-3.5 Turbo
    )
    
    # Display model information
    st.sidebar.markdown(f"**Selected Model:** {selected_qa_model}")
    st.sidebar.markdown("**Note:** The embedding model is fixed to the one used when building the tree")
      # Verify tree path if not already verified
    if 'tree_path' not in st.session_state:
        with st.spinner("Verifying RAPTOR tree file..."):
            try:
                verified_tree_path = load_tree(tree_path)
                if verified_tree_path:
                    st.session_state.tree_path = verified_tree_path
                    st.sidebar.success("RAPTOR tree file verified successfully!")
                else:
                    st.sidebar.error("Failed to verify RAPTOR tree file")
                    return
            except Exception as e:
                st.sidebar.error(f"Error verifying tree file: {e}")
                return
      # Reset button
    if st.sidebar.button("Reset Conversation"):
        st.session_state.conversation = []
        st.session_state.ra = None
        st.session_state.qa_model = None  # Reset the QA model selection too
        st.success("Conversation has been reset!")
    
    # Initialize conversation history if needed
    if 'conversation' not in st.session_state:
        st.session_state.conversation = []
    
    # Display conversation history
    for entry in st.session_state.conversation:
        question = entry.get("question", "")
        answer = entry.get("answer", "")
        context = entry.get("context", "")
        
        st.markdown(f"**Question:** {question}")
        render_answer(answer, context)
        st.markdown("---")
    
    # Question input
    question = st.text_area("Enter your question:", height=100)
    
    if st.button("Get Answer"):
        if not question:
            st.warning("Please enter a question.")
            return
              # Initialize or update RA if QA model has changed
        if ('ra' not in st.session_state or 
            'qa_model' not in st.session_state or 
            st.session_state.qa_model != selected_qa_model):
            
            with st.spinner(f"Initializing with {selected_qa_model}..."):
                ra = initialize_retrieval_augmentation(
                    st.session_state.tree_path, 
                    selected_qa_model
                )
                st.session_state.ra = ra
                st.session_state.qa_model = selected_qa_model
        
        # Process the question
        with st.spinner("Processing your question..."):
            try:
                answer, layer_info = st.session_state.ra.answer_question(
                    question=question,
                    return_layer_information=True
                )
                
                # Add to conversation history
                st.session_state.conversation.append({
                    "question": question,
                    "answer": answer,
                    "context": layer_info.get("context", "")
                })
                
                # Force refresh
                st.experimental_rerun()
                
            except Exception as e:
                st.error(f"Error processing question: {str(e)}")
                logging.error(f"Error processing question: {e}", exc_info=True)

def main():
    """Main entry point for the Streamlit app"""
    # Load configuration
    config = load_config()
    if not config:
        st.error("Failed to load configuration. Please check your config.yaml file.")
        return
    
    # Get tree path from config
    tree_path = config['raptor']['tree']['save_path']
    if not tree_path:
        st.error("Tree path not found in configuration. Please check your config.yaml file.")
        return
    
    # Display the interface
    display_interface(tree_path)

if __name__ == "__main__":
    main()
