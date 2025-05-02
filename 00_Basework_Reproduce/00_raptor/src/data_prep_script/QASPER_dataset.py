"""
QASPER Dataset Processor for the RAPTOR pipeline.
(Recursive Abstractive Processing for Tree-Organized Retrieval)

This script downloads and processes the QASPER dataset into a format suitable for
use with the RAPTOR pipeline.
"""
import json
import os
import requests
import tarfile
import io
import time
import yaml
import argparse
from pathlib import Path
from tqdm import tqdm
from datetime import datetime

def load_config(config_path="config.yaml"):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def create_directories(config):
    """Create necessary directories for the dataset"""
    os.makedirs(config['dataset']['qasper']['papers_dir'], exist_ok=True)
    os.makedirs(config['dataset']['qasper']['qas_dir'], exist_ok=True)
    os.makedirs(config['raptor']['tree']['output_dir'], exist_ok=True)

def download_and_extract_data(url, temp_dir):
    """Download and extract the dataset files from URL with progress bar"""
    print(f"Downloading from {url}...")
    
    start_time = time.time()
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    # Get file size for progress bar
    total_size = int(response.headers.get('content-length', 0))
    downloaded = 0
    
    # Download with progress bar
    content = bytearray()
    with tqdm(total=total_size, unit='B', unit_scale=True, desc="Downloading") as pbar:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                content.extend(chunk)
                downloaded += len(chunk)
                pbar.update(len(chunk))
    
    # Extract the tarfile with progress bar
    print("Extracting files...")
    tar = tarfile.open(fileobj=io.BytesIO(content))
    
    members = tar.getmembers()
    with tqdm(total=len(members), desc="Extracting") as pbar:
        for member in members:
            tar.extract(member, path=temp_dir)
            pbar.update(1)
    
    tar.close()
    
    elapsed = time.time() - start_time
    print(f"Download and extraction completed in {elapsed:.2f} seconds")
    
    return True

def read_json_file(file_path):
    """Read a JSON file and return its contents"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON file {file_path}: {e}")
        return {}

def process_qasper_file(file_path, split_name, config):
    """Process a Qasper JSON file and extract papers and QA pairs"""
    print(f"Processing {split_name} data from {file_path}...")
    data = read_json_file(file_path)
    
    if not data:
        print(f"Warning: No data found in {file_path}")
        return 0, 0
    
    paper_count = 0
    qa_count = 0
    papers_dir = config['dataset']['qasper']['papers_dir']
    qas_dir = config['dataset']['qasper']['qas_dir']
    
    # Process each paper with improved progress tracking
    for paper_id, paper_data in tqdm(data.items(), desc=f"Processing {split_name} papers"):
        try:
            # Add ID to the paper data
            paper_data["id"] = paper_id
            
            # Extract paper content
            paper_content = {
                "id": paper_id,
                "title": paper_data.get("title", ""),
                "abstract": paper_data.get("abstract", ""),
                "full_text": paper_data.get("full_text", []),
                "split": split_name  # Add which split this paper belongs to
            }
            
            # Save paper content as JSON
            paper_json_path = os.path.join(papers_dir, f"paper_{paper_id}.json")
            with open(paper_json_path, "w", encoding='utf-8') as f:
                json.dump(paper_content, f, indent=2)
            
            paper_count += 1
            
            # Extract QA pairs with more structured approach
            qa_pairs = []
            for qa in paper_data.get("qas", []):
                question = qa.get("question", "")
                question_id = qa.get("question_id", "")
                
                # Skip empty questions
                if not question.strip():
                    continue
                
                # Process each answer for this question
                for answer_obj in qa.get("answers", []):
                    answer = answer_obj.get("answer", {})
                    
                    qa_pair = {
                        "paper_id": paper_id,
                        "question_id": question_id,
                        "question": question,
                        "nlp_background": qa.get("nlp_background", ""),
                        "topic_background": qa.get("topic_background", ""),
                        "paper_read": qa.get("paper_read", ""),
                        "unanswerable": answer.get("unanswerable", False),
                        "extractive_spans": answer.get("extractive_spans", []),
                        "yes_no": answer.get("yes_no", None),
                        "free_form_answer": answer.get("free_form_answer", ""),
                        "evidence": answer.get("evidence", []),
                        "highlighted_evidence": answer.get("highlighted_evidence", []),
                        "annotation_id": answer_obj.get("annotation_id", ""),
                        "worker_id": answer_obj.get("worker_id", ""),
                        "split": split_name  # Add split information
                    }
                    
                    qa_pairs.append(qa_pair)
                    qa_count += 1
            
            # Save QA pairs for this paper
            qa_json_path = os.path.join(qas_dir, f"qa_for_paper_{paper_id}.json")
            with open(qa_json_path, "w", encoding='utf-8') as f:
                json.dump(qa_pairs, f, indent=2)
                
        except Exception as e:
            print(f"Error processing paper {paper_id}: {e}")
            continue
    
    return paper_count, qa_count

def create_text_corpus(config, paper_stats):
    """Create plain text corpus files for each paper"""
    print("Creating plain text corpus files...")
    papers_dir = config['dataset']['qasper']['papers_dir']
    papers_processed = 0
    
    # Get all JSON files in papers directory
    json_files = [f for f in os.listdir(papers_dir) if f.endswith('.json')]
    
    for filename in tqdm(json_files, desc="Creating text corpus"):
        try:
            paper_id = filename.replace("paper_", "").replace(".json", "")
            paper_json_path = os.path.join(papers_dir, filename)
            
            with open(paper_json_path, "r", encoding='utf-8') as f:
                paper_data = json.load(f)
            
            # Create a text file with all paper content
            text_content = f"Title: {paper_data['title']}\n\n"
            text_content += f"Abstract: {paper_data['abstract']}\n\n"
            
            for section in paper_data.get("full_text", []):
                section_name = section.get("section_name", "")
                text_content += f"Section: {section_name}\n"
                
                for paragraph in section.get("paragraphs", []):
                    text_content += f"{paragraph}\n\n"
            
            paper_txt_path = os.path.join(papers_dir, f"paper_{paper_id}.txt")
            with open(paper_txt_path, "w", encoding='utf-8') as f:
                f.write(text_content)
            
            papers_processed += 1
            
            # Record section and paragraph counts for statistics
            section_count = len(paper_data.get("full_text", []))
            paragraph_count = sum(len(section.get("paragraphs", [])) for section in paper_data.get("full_text", []))
            
            # Update paper stats
            paper_stats[paper_id] = {
                "title": paper_data['title'],
                "section_count": section_count,
                "paragraph_count": paragraph_count,
                "split": paper_data.get("split", "unknown")
            }
            
        except Exception as e:
            print(f"Error creating text corpus for {filename}: {e}")
    
    return papers_processed, paper_stats

def generate_report(config, stats):
    """Generate a detailed report about the dataset extraction"""
    if not stats or not stats.get("total_papers"):
        print("No statistics available for report generation")
        return
        
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    report = f"""# QASPER Dataset Extraction Report
Generated: {timestamp}

## Overview
- Total papers processed: {stats['total_papers']}
- Total QA pairs extracted: {stats['total_qa_pairs']}

## Split Distribution
- Train split: {stats['split_stats'].get('train', {}).get('papers', 0)} papers, {stats['split_stats'].get('train', {}).get('qa_pairs', 0)} QA pairs
- Validation split: {stats['split_stats'].get('validation', {}).get('papers', 0)} papers, {stats['split_stats'].get('validation', {}).get('qa_pairs', 0)} QA pairs
- Test split: {stats['split_stats'].get('test', {}).get('papers', 0)} papers, {stats['split_stats'].get('test', {}).get('qa_pairs', 0)} QA pairs

## Document Statistics
- Average sections per paper: {stats.get('avg_sections_per_paper', 0):.2f}
- Average paragraphs per paper: {stats.get('avg_paragraphs_per_paper', 0):.2f}
- Average QA pairs per paper: {stats.get('avg_qa_per_paper', 0):.2f}

## Using with RAPTOR Pipeline
The QASPER dataset is now ready to be used with the RAPTOR (Recursive Abstractive Processing for Tree-Organized Retrieval) pipeline.

### Dataset Structure
- Papers are stored as both JSON and plain text in `{config['dataset']['qasper']['papers_dir']}`
- QA pairs are stored in `{config['dataset']['qasper']['qas_dir']}`
- A merged file with all QA pairs is available at `{os.path.join(config['dataset']['qasper']['qas_dir'], 'all_qa_pairs.json')}`

### Next Steps
1. Build the hierarchical tree structure using `tree_builder.py`
2. Create embeddings for each node in the tree
3. Run retrieval experiments using `tree_retriever.py`
4. Use `pipeline_for_QA.py` to evaluate QA performance

Run the RAPTOR pipeline with:
```bash
python run_rag_pipeline.py --dataset qasper --model gpt-3.5-turbo
```

For more information, see the RAPTOR documentation.
"""

    # Save report if configured
    if config.get('reporting', {}).get('save_report'):
        report_file = config['reporting']['report_file']
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report)
        print(f"Report saved to {report_file}")
    
    return report

def main(config_path="config.yaml"):
    """Main execution function"""
    start_time = time.time()
    
    # Load configuration
    config = load_config(config_path)
    
    # Create necessary directories
    create_directories(config)
    
    # Initialize statistics
    stats = {
        "total_papers": 0,
        "total_qa_pairs": 0,
        "paper_stats": {},
        "split_stats": {
            "train": {"papers": 0, "qa_pairs": 0},
            "validation": {"papers": 0, "qa_pairs": 0},
            "test": {"papers": 0, "qa_pairs": 0}
        }
    }
    
    try:
        # Download and extract datasets
        temp_dir = config['dataset']['qasper']['temp_dir']
        download_and_extract_data(config['dataset']['qasper']['url_train_dev'], temp_dir)
        download_and_extract_data(config['dataset']['qasper']['url_test'], temp_dir)
        
        # Process each file
        train_file = os.path.join(temp_dir, config['dataset']['qasper']['train_file'])
        dev_file = os.path.join(temp_dir, config['dataset']['qasper']['dev_file'])
        test_file = os.path.join(temp_dir, config['dataset']['qasper']['test_file'])
        
        # Process train file
        train_papers, train_qa_pairs = process_qasper_file(train_file, "train", config)
        stats["total_papers"] += train_papers
        stats["total_qa_pairs"] += train_qa_pairs
        stats["split_stats"]["train"]["papers"] = train_papers
        stats["split_stats"]["train"]["qa_pairs"] = train_qa_pairs
        
        # Process dev file
        dev_papers, dev_qa_pairs = process_qasper_file(dev_file, "validation", config)
        stats["total_papers"] += dev_papers
        stats["total_qa_pairs"] += dev_qa_pairs
        stats["split_stats"]["validation"]["papers"] = dev_papers
        stats["split_stats"]["validation"]["qa_pairs"] = dev_qa_pairs
        
        # Process test file
        test_papers, test_qa_pairs = process_qasper_file(test_file, "test", config)
        stats["total_papers"] += test_papers
        stats["total_qa_pairs"] += test_qa_pairs
        stats["split_stats"]["test"]["papers"] = test_papers
        stats["split_stats"]["test"]["qa_pairs"] = test_qa_pairs
    
        # Create a merged QA file with all questions and answers
        print("Creating merged QA file...")
        all_qa_pairs = []
        qas_dir = config['dataset']['qasper']['qas_dir']
        
        for filename in os.listdir(qas_dir):
            if filename.endswith(".json") and not filename == "all_qa_pairs.json":
                try:
                    qa_json_path = os.path.join(qas_dir, filename)
                    with open(qa_json_path, "r", encoding='utf-8') as f:
                        qa_pairs = json.load(f)
                        all_qa_pairs.extend(qa_pairs)
                except Exception as e:
                    print(f"Error processing QA file {filename}: {e}")
        
        all_qa_path = os.path.join(qas_dir, "all_qa_pairs.json")
        with open(all_qa_path, "w", encoding='utf-8') as f:
            json.dump(all_qa_pairs, f, indent=2)
        
        print(f"Total QA pairs in merged file: {len(all_qa_pairs)}")
        
        # Create text corpus and collect stats
        papers_processed, paper_stats = create_text_corpus(config, {})
        stats["paper_stats"] = paper_stats
        
        # Calculate aggregate statistics
        if stats["paper_stats"]:
            stats["avg_sections_per_paper"] = sum(p.get("section_count", 0) for p in stats["paper_stats"].values()) / max(1, len(stats["paper_stats"]))
            stats["avg_paragraphs_per_paper"] = sum(p.get("paragraph_count", 0) for p in stats["paper_stats"].values()) / max(1, len(stats["paper_stats"]))
        
        if stats["total_papers"] > 0:
            stats["avg_qa_per_paper"] = stats["total_qa_pairs"] / stats["total_papers"]
            
        # Clean up temporary files
        print("Cleanup temporary files...")
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)
        
        # Generate and display the report
        elapsed = time.time() - start_time
        print(f"\nExtraction completed in {elapsed:.2f} seconds!")
        print(f"Total papers: {stats['total_papers']}, Total QA pairs: {stats['total_qa_pairs']}")
        
        report = generate_report(config, stats)
        print("\n" + "="*50 + "\nEXTRACTION SUMMARY\n" + "="*50)
        print(f"Total papers processed: {stats['total_papers']}")
        print(f"Total QA pairs extracted: {stats['total_qa_pairs']}")
        print(f"Train split: {stats['split_stats']['train']['papers']} papers, {stats['split_stats']['train']['qa_pairs']} QA pairs")
        print(f"Validation split: {stats['split_stats']['validation']['papers']} papers, {stats['split_stats']['validation']['qa_pairs']} QA pairs")
        print(f"Test split: {stats['split_stats']['test']['papers']} papers, {stats['split_stats']['test']['qa_pairs']} QA pairs")
        print("="*50)
        
        print("\nAll done! The dataset is ready to be used with the RAPTOR pipeline.")
        if config.get('reporting', {}).get('save_report'):
            print(f"See detailed report at {config['reporting']['report_file']}")
    
    except Exception as e:
        print(f"Error during processing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="QASPER Dataset Processor for RAPTOR pipeline")
    parser.add_argument("--config", default="config.yaml", help="Path to config file")
    args = parser.parse_args()
    
    main(args.config)