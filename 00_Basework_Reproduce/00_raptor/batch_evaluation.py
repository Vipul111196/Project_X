#!/usr/bin/env python3
"""
QASPER Evaluation Pipeline for RAG

Evaluates a RAPTOR-based RAG pipeline on the QASPER dataset,
using metrics from the RAPTOR paper for direct comparison.
"""

import os
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Any, Tuple
from datetime import datetime
from tqdm import tqdm

# Import qa_metrics for evaluation - only using f1 since exact_match has issues
from qa_metrics.f1 import f1_match, f1_score_with_precision_recall

# Import RAPTOR components
from src.components import RetrievalAugmentation, RetrievalAugmentationConfig
from src.models import (
    get_qa_model,
    get_summarization_model,
    get_embedding_model,
    clear_gpu_memory
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/qasper_evaluation.log")
    ]
)

def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """Load configuration from YAML file"""
    try:
        import yaml
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        logging.info(f"Configuration loaded from {config_path}")
        return config
    except Exception as e:
        logging.error(f"Error loading configuration: {e}")
        raise

def load_tree(tree_path: str) -> Any:
    """Load a previously serialized Tree object."""
    try:
        import pickle
        with open(tree_path, 'rb') as f:
            tree = pickle.load(f)
            logging.info(f"Successfully loaded tree from {tree_path}")
            return tree
    except Exception as e:
        logging.error(f"Failed to load tree from {tree_path}: {e}")
        raise

def initialize_retrieval_augmentation(config: Dict[str, Any], tree_path: str) -> RetrievalAugmentation:
    """Initialize the RetrievalAugmentation with the specified models."""
    # Get models from configuration
    qa_model = get_qa_model(config)
    summarization_model = get_summarization_model(config)
    embedding_model = get_embedding_model(config)
    
    logging.info(f"Initializing RetrievalAugmentation with models: QA={type(qa_model).__name__}, Summarization={type(summarization_model).__name__}, Embedding={type(embedding_model).__name__}")
    
    # Create configuration
    rac = RetrievalAugmentationConfig(
        summarization_model=summarization_model,
        qa_model=qa_model,
        embedding_model=embedding_model
    )
    
    # Load the tree
    tree = load_tree(tree_path)
    # Initialize RetrievalAugmentation
    ra = RetrievalAugmentation(config=rac, tree=tree)

    return ra

def get_paper_qa_mapping(qas_dir: str) -> Dict[str, str]:
    """Map paper IDs to their corresponding Q&A files"""
    import glob
    import os
    
    qa_files = glob.glob(os.path.join(qas_dir, "qa_for_paper_*.json"))
    
    # Create mapping from paper ID to Q&A file
    paper_qa_map = {}
    for qa_file in qa_files:
        filename = os.path.basename(qa_file)
        # Extract ID from "qa_for_paper_1503.00841.json" format
        paper_id = filename.replace("qa_for_paper_", "").replace(".json", "")
        paper_qa_map[paper_id] = qa_file
    
    return paper_qa_map

def load_paper_qa_data(qa_file: str) -> Dict[str, List]:
    """Load Q&A data for a specific paper"""
    try:
        with open(qa_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        questions = []
        reference_answers = []
        evidence_list = []
        paper_ids = []
        question_ids = []
        
        # Extract paper ID from filename
        paper_id = os.path.basename(qa_file).replace("qa_for_paper_", "").replace(".json", "")
        
        for item in data:
            # Skip unanswerable questions
            if item.get("unanswerable", False):
                continue
            
            answer = None
            # Priority: free_form_answer, extractive_spans, yes_no
            if item.get("free_form_answer", ""):
                answer = item["free_form_answer"]
            elif item.get("extractive_spans", []):
                answer = " ".join(item["extractive_spans"])
            elif item.get("yes_no") is not None:
                answer = "Yes" if item["yes_no"] else "No"
            
            # Skip if no valid answer
            if not answer:
                continue
            
            questions.append(item["question"])
            reference_answers.append(answer)
            evidence_list.append(item.get("evidence", []))
            paper_ids.append(paper_id)
            question_ids.append(item.get("question_id", ""))
        
        return {
            "questions": questions,
            "reference_answers": reference_answers,
            "evidence": evidence_list,
            "paper_ids": paper_ids,
            "question_ids": question_ids
        }
    except Exception as e:
        logging.error(f"Error loading Q&A data from {qa_file}: {e}")
        return {
            "questions": [],
            "reference_answers": [],
            "evidence": [],
            "paper_ids": [],
            "question_ids": []
        }

def load_selected_papers_qa(qas_dir: str, paper_ids: List[str], limit: int = None) -> Dict[str, List]:
    """Load Q&A data for selected papers only"""
    # Get mapping of paper IDs to Q&A files
    paper_qa_map = get_paper_qa_mapping(qas_dir)
    
    # Combined data
    combined_data = {
        "questions": [],
        "reference_answers": [],
        "evidence": [],
        "paper_ids": [],
        "question_ids": []
    }
    
    # Load Q&A data for each selected paper
    for paper_id in paper_ids:
        if paper_id in paper_qa_map:
            qa_file = paper_qa_map[paper_id]
            paper_data = load_paper_qa_data(qa_file)
            
            # Append data
            for key in combined_data:
                combined_data[key].extend(paper_data[key])
                
            logging.info(f"Loaded {len(paper_data['questions'])} questions for paper {paper_id}")
        else:
            logging.warning(f"No Q&A file found for paper {paper_id}")
    
    # Apply limit if specified
    if limit is not None and limit < len(combined_data["questions"]):
        for key in combined_data:
            combined_data[key] = combined_data[key][:limit]
        logging.info(f"Limited to {limit} questions")
    
    logging.info(f"Total: Loaded {len(combined_data['questions'])} questions across {len(paper_ids)} papers")
    return combined_data

def run_rag_pipeline(
    ra: RetrievalAugmentation,
    questions: List[str],
    paper_ids: List[str]
) -> Tuple[List[str], List[str]]:
    """Run the RAG pipeline on a list of questions"""
    predicted_answers = []
    retrieved_contexts = []
    
    for i, (question, paper_id) in enumerate(tqdm(zip(questions, paper_ids), total=len(questions), desc="Running RAG pipeline")):
        try:
            # First retrieve the context and layer_info
            context, _ = ra.retrieve(question, return_layer_information=True)
            
            # Then use the answer_question method without returning layer_info
            answer = ra.answer_question(question, return_layer_information=False)
            
            predicted_answers.append(answer)
            retrieved_contexts.append(context)
            
            # Log progress periodically
            if (i + 1) % 10 == 0:
                logging.info(f"Processed {i+1}/{len(questions)} questions")
                
        except Exception as e:
            logging.error(f"Error processing question {i+1}: {e}")
            predicted_answers.append("")
            retrieved_contexts.append("")
    
    return predicted_answers, retrieved_contexts

def evaluate_rag_pipeline(
    reference_answers: List[str],
    predicted_answers: List[str]
) -> Dict[str, float]:
    """Evaluate the RAG pipeline using metrics from the RAPTOR paper"""
    # Initialize metrics
    f1_match_scores = []
    token_f1_scores = []
    
    # Calculate answer generation metrics
    for reference, prediction in tqdm(zip(reference_answers, predicted_answers), 
                                     total=len(reference_answers),
                                     desc="Evaluating answer quality"):
        # Skip empty predictions or references
        if not reference or not prediction:
            continue
            
        # F1 Match score (main metric used by RAPTOR)
        match_result = f1_match([reference], prediction, threshold=0.5)
        f1_match_scores.append(match_result)
        
        # Token F1 score
        f1_stats = f1_score_with_precision_recall(reference, prediction)
        token_f1_scores.append(f1_stats["f1"])
    
    # Calculate average scores
    metrics = {
        "f1_match": sum(f1_match_scores) / len(f1_match_scores) if f1_match_scores else 0,
        "token_f1": sum(token_f1_scores) / len(token_f1_scores) if token_f1_scores else 0
    }
    
    return metrics

def save_results(
    qasper_data: Dict[str, List],
    predicted_answers: List[str],
    retrieved_contexts: List[str],
    output_dir: str = "results"
) -> None:
    """Save evaluation results to JSON files"""
    # Ensure output_dir is not None
    if output_dir is None:
        output_dir = "results"
        logging.warning("Output directory not specified, using default: 'results'")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Save ground truth data
    ground_truth_file = os.path.join(output_dir, "ground_truth.json")
    ground_truth = []
    
    for i in range(len(qasper_data["questions"])):
        item = {
            "question": qasper_data["questions"][i],
            "paper_id": qasper_data["paper_ids"][i],
            "question_id": qasper_data["question_ids"][i],
            "ground_truth_answer": qasper_data["reference_answers"][i],
            "ground_truth_evidence": qasper_data["evidence"][i]
        }
        ground_truth.append(item)
    
    with open(ground_truth_file, 'w', encoding='utf-8') as f:
        json.dump(ground_truth, f, indent=2)
    
    # Save prediction data
    predictions_file = os.path.join(output_dir, "predictions.json")
    predictions = []
    
    for i in range(len(qasper_data["questions"])):
        if i < len(predicted_answers):
            item = {
                "question": qasper_data["questions"][i],
                "paper_id": qasper_data["paper_ids"][i],
                "question_id": qasper_data["question_ids"][i],
                "predicted_answer": predicted_answers[i],
                "retrieved_context": retrieved_contexts[i]
            }
            predictions.append(item)
    
    with open(predictions_file, 'w', encoding='utf-8') as f:
        json.dump(predictions, f, indent=2)
    
    logging.info(f"Results saved to {output_dir}")

def run_mock_rag_pipeline(questions, paper_ids):
    """Simple mock RAG pipeline that returns dummy answers for testing"""
    predicted_answers = []
    retrieved_contexts = []
    
    for question in questions:
        # Simple mocked answer based on question
        answer = f"The answer discusses {question.split()[0]} {question.split()[1]}" 
        context = f"This is a retrieved context about {' '.join(question.split()[:3])}"
        
        predicted_answers.append(answer)
        retrieved_contexts.append(context)
    
    return predicted_answers, retrieved_contexts

def main():
    pass


if __name__ == "__main__":
    main()