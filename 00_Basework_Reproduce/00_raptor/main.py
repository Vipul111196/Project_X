"""
RAPTOR Tree Generator

This script builds the hierarchical tree structure for the Recursive Abstractive 
Processing for Tree-Organized Retrieval (RAPTOR) system.

It reads documents, creates embeddings, and builds a tree structure that can be used
for hierarchical retrieval.
"""
import os
import glob
import logging
import yaml
import argparse
import sys
from typing import Dict, Any, List
from pathlib import Path

# Import the components
from src.components import RetrievalAugmentation, RetrievalAugmentationConfig
from src.models import (
    get_qa_model, 
    get_summarization_model, 
    get_unified_model, 
    get_embedding_model, 
    clear_gpu_memory
)

def setup_logging(config: Dict[str, Any]) -> None:
    """
    Set up logging based on configuration
    
    Args:
        config: Configuration dictionary with logging settings
    """
    log_level = getattr(logging, config["logging"]["log_level"])
    log_file = config["logging"]["log_file"]
    
    # Create logs directory if it doesn't exist
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    logging.info("RAPTOR Tree Generator started")


def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """
    Load configuration from YAML file
    
    Args:
        config_path: Path to the YAML configuration file
        
    Returns:
        Dict: Configuration dictionary
    """
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        logging.info(f"Configuration loaded from {config_path}")
        return config
    except Exception as e:
        logging.error(f"Error loading configuration: {e}")
        sys.exit(1)


def load_papers(config: Dict[str, Any], num_papers: int = None) -> str:
    """
    Load papers from the dataset directory based on configuration
    
    Args:
        config: Configuration dictionary
        num_papers: Optional number of papers to load
        
    Returns:
        str: Combined text of loaded papers
    """
    paper_dir = config["dataset"]["qasper"]["papers_dir"]
    sub_num_papers = config["dataset"]["subset"]["papers_range"] # reading a subset of papers
    # sub_num_papers = list(map(int, sub_num_papers.split(",")))

    # Load all papers from directory
    paper_files = glob.glob(os.path.join(paper_dir, "*.txt"))
    
    # Select subset of papers if specified
    if sub_num_papers:
        paper_files = paper_files[sub_num_papers[0]:sub_num_papers[1]]
        logging.info(f"Loading {len(paper_files)} papers")
    else:
        logging.info(f"Loading all {len(paper_files)} papers")
    all_papers_text = ""
    paper_count = 0
    
    for paper_file in paper_files:
        try:
            with open(paper_file, 'r', encoding='utf-8') as file:
                print(f"Loading paper: {os.path.basename(paper_file)}")
                paper_text = file.read()
                all_papers_text += f"\n\n{'='*50}\nPAPER: {os.path.basename(paper_file)}\n{'='*50}\n\n"
                all_papers_text += paper_text
                paper_count += 1
                logging.info(f"Loaded paper: {os.path.basename(paper_file)}")
        except Exception as e:
            logging.error(f"Error loading paper {paper_file}: {e}")
    
    logging.info(f"Loaded {paper_count} papers with total characters: {len(all_papers_text)}")
    return all_papers_text

def initialize_models(config: Dict[str, Any]):
    """
    Initialize models based on configuration
    
    Args:
        config: Configuration dictionary
        
    Returns:
        tuple: (qa_model, summarization_model, unified_model, embedding_model)
    """
    # Clear GPU cache before initializing models
    clear_gpu_memory()
    
    # Try to get a unified model first
    unified_model = get_unified_model(config, )
    
    # If using unified model, it will serve as both QA and summarization model
    if unified_model:
        qa_model = unified_model
        summarization_model = unified_model
        logging.info(f"Using unified model for both QA and summarization")
    else:
        # Otherwise, get separate models
        qa_model = get_qa_model(config)
        summarization_model = get_summarization_model(config)
        logging.info("Using separate models for QA and summarization")
    
    # Get embedding model
    embedding_model = get_embedding_model(config)
    
    return qa_model, summarization_model, unified_model, embedding_model


def build_tree(config: Dict[str, Any], qa_model, summarization_model, embedding_model) -> None:
    """
    Build the RAPTOR tree from papers
    
    Args:
        config: Configuration dictionary
        qa_model: Question answering model
        summarization_model: Summarization model
        embedding_model: Embedding model
    """
    # Initialize RAG configuration
    rag_config = RetrievalAugmentationConfig(
        summarization_model=summarization_model,
        qa_model=qa_model,
        embedding_model=embedding_model
    )
    
    # Initialize RAG
    logging.info("Initializing RAPTOR pipeline...")
    RA = RetrievalAugmentation(config=rag_config)
    logging.info("Retrieval-Augmentation pipeline initialized")
    
    # Load papers
    all_papers_text = load_papers(config)
    
    # Add all papers at once to RAG
    logging.info("Adding all papers to the RAG pipeline...")
    RA.add_documents(all_papers_text)
    logging.info("All papers added successfully")
    
    # Save the RAG Tree
    save_path = config["raptor"]["tree"]["save_path"]
    logging.info(f"Saving RAG tree to {save_path}...")
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    RA.save(save_path)
    logging.info("RAG tree saved successfully")
    
    # Clear memory after building tree
    clear_gpu_memory()


def main(config_path: str = "config.yaml") -> None:
    """
    Main function to run the tree generator
    
    Args:
        config_path: Path to the configuration file
    """
    # Load configuration
    config = load_config(config_path)
    
    # Set up logging
    setup_logging(config)
    
    try:        
        # Initialize models based on configuration
        qa_model, summarization_model, unified_model, embedding_model = initialize_models(config)
        
        # Build tree using the models
        build_tree(config, qa_model, summarization_model, embedding_model)
        
        logging.info("RAPTOR tree generation completed successfully")
        
    except Exception as e:
        logging.error(f"Error in tree generator: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RAPTOR Tree Generator")
    parser.add_argument("--config", default="config.yaml", help="Path to configuration file")
    args = parser.parse_args()
    
    main(args.config)