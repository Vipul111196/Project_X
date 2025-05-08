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

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Evaluate RAPTOR on QASPER dataset")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    parser.add_argument("--tree", type=str, help="Path to RAPTOR tree file")
    parser.add_argument("--qas-dir", type=str, help="Directory containing Q&A files")
    parser.add_argument("--papers-dir", type=str, help="Directory containing paper files")
    parser.add_argument("--output-dir", type=str, help="Output directory for results")
    parser.add_argument("--papers", type=str, help="Comma-separated list of paper IDs to include")
    parser.add_argument("--limit", type=int, help="Limit number of questions to process (for testing)")
    parser.add_argument("--test", action="store_true", help="Run in test mode with simplified pipeline")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    
    return parser.parse_args()

def main():
    """Main function to run the evaluation pipeline"""
    # Parse arguments
    args = parse_arguments()
    
    # Create logs directory
    os.makedirs("logs", exist_ok=True)
    
    try:
        # Load configuration
        config = load_config(args.config)
        
        # Set default values from config (if available)
        tree_path = args.tree or config.get("raptor", {}).get("tree", {}).get("save_path")
        qas_dir = args.qas_dir or config.get("evaluation", {}).get("qas_dir")
        papers_dir = args.papers_dir or config.get("evaluation", {}).get("papers_dir")
        output_dir = args.output_dir or config.get("evaluation", {}).get("output_dir", "results")
        
        # Ensure we have a default output_dir
        if output_dir is None:
            output_dir = "results"
            logging.info(f"Using default output directory: {output_dir}")
        
        # Get paper IDs from command line or config
        paper_ids = None
        if args.papers:
            paper_ids = [pid.strip() for pid in args.papers.split(",")]
        elif config.get("evaluation", {}).get("papers"):
            # Handle both string and list format in config
            papers_config = config["evaluation"]["papers"]
            if isinstance(papers_config, str):
                paper_ids = [pid.strip() for pid in papers_config.split(",")]
            else:
                paper_ids = papers_config
        
        # Validate required parameters
        if not tree_path and not args.test:
            logging.error("Tree path not specified. Use --tree or add to config.yaml")
            return
            
        if not qas_dir:
            logging.error("QA directory not specified. Use --qas-dir or add to config.yaml")
            return
            
        if not paper_ids:
            logging.error("No paper IDs specified. Use --papers or add to config.yaml")
            return
            
        logging.info(f"Using papers: {paper_ids}")
        logging.info(f"QA directory: {qas_dir}")
        if papers_dir:
            logging.info(f"Papers directory: {papers_dir}")
        logging.info(f"Output directory: {output_dir}")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Load Q&A data for selected papers
        qasper_data = load_selected_papers_qa(qas_dir, paper_ids, args.limit)
        
        if not qasper_data["questions"]:
            logging.error("No questions found for the specified papers")
            return
        
        # Run RAG pipeline
        if args.test:
            logging.info("Running in test mode with mock RAG pipeline")
            predicted_answers, retrieved_contexts = run_mock_rag_pipeline(
                qasper_data["questions"],
                qasper_data["paper_ids"]
            )
        else:
            # Initialize RAPTOR
            ra = initialize_retrieval_augmentation(config, tree_path)
            
            # Add debugging for the first question to understand layer_info structure
            if args.verbose and qasper_data["questions"]:
                test_question = qasper_data["questions"][0]
                logging.info(f"DEBUG: Testing layer_info structure with question: {test_question}")
                try:
                    test_answer, test_layer_info = ra.answer_question(
                        question=test_question,
                        return_layer_information=True
                    )
                    logging.info(f"DEBUG: layer_info type: {type(test_layer_info)}")
                    if isinstance(test_layer_info, list):
                        logging.info(f"DEBUG: layer_info is a list with {len(test_layer_info)} items")
                        if test_layer_info and len(test_layer_info) > 0:
                            sample_item = test_layer_info[0]
                            logging.info(f"DEBUG: First item type: {type(sample_item)}")
                            if isinstance(sample_item, dict):
                                logging.info(f"DEBUG: First item keys: {list(sample_item.keys())}")
                    elif isinstance(test_layer_info, dict):
                        logging.info(f"DEBUG: layer_info keys: {list(test_layer_info.keys())}")
                except Exception as e:
                    logging.error(f"DEBUG: Error in test question: {e}")
            
            # Run RAG pipeline on QASPER questions
            predicted_answers, retrieved_contexts = run_rag_pipeline(
                ra, 
                qasper_data["questions"],
                qasper_data["paper_ids"]
            )
            
            # Check if retrieved_contexts are empty
            if predicted_answers and all(not context for context in retrieved_contexts):
                logging.warning("All retrieved contexts are empty. This may indicate an issue with context extraction.")
        
        # Evaluate the pipeline
        metrics = evaluate_rag_pipeline(
            qasper_data["reference_answers"],
            predicted_answers
        )
        
        # Print verbose output if requested
        if args.verbose:
            for i, (question, ref_answer, pred_answer) in enumerate(zip(
                qasper_data["questions"][:5], 
                qasper_data["reference_answers"][:5], 
                predicted_answers[:5]
            )):
                print(f"\nQuestion {i+1}: {question}")
                print(f"Reference answer: {ref_answer}")
                print(f"Predicted answer: {pred_answer}")
                
                # Calculate metrics for this question
                f1 = f1_score_with_precision_recall(ref_answer, pred_answer)["f1"]
                print(f"F1 score: {f1:.4f}")
        
        # Print results
        print("\nRAG Pipeline Evaluation Results:")
        print(f"F1 Match Score: {metrics['f1_match'] * 100:.2f}%")
        print(f"Token F1 Score: {metrics['token_f1'] * 100:.2f}%")
        
        # Compare with RAPTOR results
        print("\nComparison with RAPTOR:")
        print("Your RAG Pipeline F1 Match Score: {:.2f}%".format(metrics['f1_match'] * 100))
        print("RAPTOR (GPT-3) F1 Match Score: 53.10%")
        print("RAPTOR (GPT-4) F1 Match Score: 55.70%")
        print("RAPTOR (UnifiedQA) F1 Match Score: 36.60%")
        
        # Save detailed results - ensure output_dir is not None
        if output_dir:
            save_results(
                qasper_data,
                predicted_answers,
                retrieved_contexts,
                output_dir
            )
        else:
            save_results(
                qasper_data,
                predicted_answers,
                retrieved_contexts
            )
        
    except Exception as e:
        logging.error(f"Error in evaluation pipeline: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Clean up
        clear_gpu_memory()

if __name__ == "__main__":
    main()