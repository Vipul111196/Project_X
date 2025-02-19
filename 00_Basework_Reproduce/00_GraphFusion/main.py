# Import necessary libraries
import os
import yaml
import json
import pickle
from tqdm import tqdm
from dotenv import load_dotenv

# Import local modules
from src.logger import my_custom_logger
from src.components.entity_extraction import extract_concepts_and_abstracts
from src.components.triplets_extraction import extract_candidate_triples
from src.components.graph_fusion import perform_knowledge_fusion
from src.components.graph_to_neo4j import build_graph_document_from_triples
from src.components.models import KnowledgeGraphLLM

def load_config(config_path="config.yaml"):
    """
    Load configuration from YAML file
    
    Args:
        config_path (str): Path to the configuration file
        
    Returns:
        dict: Loaded configuration
    """
    try:
        with open(config_path, "r") as file:
            config = yaml.safe_load(file)
        my_custom_logger.info("Configuration loaded successfully.")
        return config
    except Exception as e:
        my_custom_logger.error(f"Failed to load configuration: {e}")
        raise

def run_entity_extraction(config):
    """
    Run the entity extraction step of the pipeline
    
    Args:
        config (dict): Configuration dictionary
        
    Returns:
        tuple: Documents and concept_data
    """
    my_custom_logger.info("Starting Entity Extraction step...")
    
    # Load data
    data_path = config["step 1"]["data_path"]
    try:
        with open(data_path, "r", encoding="utf-8", errors="replace") as file:
            documents = file.readlines()
            documents = [line.strip() for line in documents]
    except Exception as e:
        my_custom_logger.error(f"Failed to load data from {data_path}: {e}")
        raise

    # Apply sampling if configured
    if config['step 1']['sample_data'].lower() == 'yes':
        sample_size = config['step 1']['sample_size']
        documents = documents[:sample_size]
        my_custom_logger.info(f"Using sample data. Number of documents: {len(documents)}")
    else:
        my_custom_logger.info(f"Using complete dataset. Number of documents: {len(documents)}")

    # Set up extraction parameters
    output_concepts_file = config["step 1"]["output_concepts_file"]
    output_concept_abstracts_json = config["step 1"]["output_concept_abstracts_json"]
    params_step_1 = config["step 1"]["params"]
    
    # Ensure output directories exist
    os.makedirs(os.path.dirname(output_concepts_file), exist_ok=True)
    os.makedirs(os.path.dirname(output_concept_abstracts_json), exist_ok=True)
    
    # Create filtered abstracts directory if specified
    filtered_abstracts_dir = params_step_1.get("filtered_abstracts_dir", "outputs/filtered_abstracts")
    os.makedirs(filtered_abstracts_dir, exist_ok=True)
    
    my_custom_logger.info(f"Output will be saved to: {output_concepts_file} and {output_concept_abstracts_json}")
    my_custom_logger.info(f"Filtered abstracts will be saved to: {filtered_abstracts_dir}")

    # Call the concept-extraction function
    try:
        extract_concepts_and_abstracts(
            documents=documents,
            output_concepts_file=output_concepts_file,
            output_concept_abstracts_json=output_concept_abstracts_json,
            logger=my_custom_logger,
            excluded_words=None,
            parameters=params_step_1
        )
        my_custom_logger.info("Concept extraction process completed successfully.")
        
        # Load extracted concept data for next step
        with open(output_concept_abstracts_json, 'r', encoding="utf-8") as f:
            concept_data = json.load(f)
        
        return documents, concept_data
    except Exception as e:
        my_custom_logger.error(f"Error in concept extraction: {e}")
        raise

def run_triplets_extraction(config, concept_data):
    """
    Run the triplets extraction step of the pipeline
    
    Args:
        config (dict): Configuration dictionary
        concept_data (dict): Concept data from previous step
        
    Returns:
        list: Extracted triples
    """
    my_custom_logger.info("Starting Triple Extraction step...")
    
    # Load API key from environment
    load_dotenv(".env")
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        my_custom_logger.warning("OPENAI_API_KEY not found in environment variables")
    
    # Set up parameters for triplet extraction
    try:
        relation_types_path = config["step 2"]["relation_types"]
        with open(relation_types_path, 'r', encoding="utf-8") as f:
            relation_types = json.load(f)
    except Exception as e:
        my_custom_logger.error(f"Failed to load relation types: {e}")
        raise
        
    # Configure language model
    llm_config = config["step 2"]["llm"]
    llm = KnowledgeGraphLLM(
        model_name=llm_config["model"], 
        max_tokens=llm_config["max_tokens"], 
        api_key=api_key
    )
    
    # Set output path
    output_triplet_path = config["step 2"]["output_triplet_path"]
    os.makedirs(os.path.dirname(output_triplet_path), exist_ok=True)
    my_custom_logger.info(f"Triples will be saved to: {output_triplet_path}")
    
    # Run triple extraction
    try:
        extract_candidate_triples(
            language_model=llm,
            output_file_path=output_triplet_path,
            relation_definitions=relation_types,
            concept_data=concept_data,
            logger=my_custom_logger,
            parameters=config["step 2"]["params"]
        )
        my_custom_logger.info(f"Triple extraction completed. Triples saved to: {output_triplet_path}")
        
        # Load the extracted triples
        with open(output_triplet_path, 'r', encoding="utf-8") as file:
            triples = [json.loads(line) for line in file]
        
        return triples
    except Exception as e:
        my_custom_logger.error(f"Error in triple extraction: {e}")
        raise

def run_graph_fusion(config, triples, relation_types, concept_data):
    """
    Run the graph fusion step of the pipeline
    
    Args:
        config (dict): Configuration dictionary
        triples (list): Extracted triples from previous step
        relation_types (dict): Relation type definitions
        concept_data (dict): Concept data
        
    Returns:
        list: Fused triples
    """
    my_custom_logger.info("Starting Knowledge Graph Fusion step...")
    
    # Set parameters for fusion
    triplets_path = config["step 3"]["triplets_path"]
    fused_triplets_path = config["step 3"]["fused_triplets_path"]
    prompt_fusion_template_path = config["step 3"]["prompt_fusion_template_path"]
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(fused_triplets_path), exist_ok=True)
    
    # Get language model from previous step
    llm_config = config["step 2"]["llm"]
    api_key = os.getenv("OPENAI_API_KEY")
    llm = KnowledgeGraphLLM(
        model_name=llm_config["model"], 
        max_tokens=llm_config["max_tokens"], 
        api_key=api_key
    )
    
    # Execute the fusion step
    try:
        fused_graph = perform_knowledge_fusion(
            language_model=llm,
            candidate_triples=triples,
            relation_definitions=relation_types,
            concept_data=concept_data,
            logger=my_custom_logger,
            prompt_fusion_path=prompt_fusion_template_path,
            fused_triplets_path=fused_triplets_path
        )
        my_custom_logger.info(f"Fusion completed. Fused triples saved to: {fused_triplets_path}")
        return fused_graph
    except Exception as e:
        my_custom_logger.error(f"Error in graph fusion: {e}")
        raise

def main():
    """
    Main function to run the GraphFusion pipeline
    """
    my_custom_logger.info("Starting GraphFusion Pipeline...")
    
    # Load configuration
    config = load_config()
    
    # Step 1: Extract Concepts and Abstracts
    documents, concept_data = run_entity_extraction(config)
    
    # Step 2: Extract Candidate Triples
    # Load relation types
    relation_types_path = config["step 2"]["relation_types"]
    with open(relation_types_path, 'r', encoding="utf-8") as f:
        relation_types = json.load(f)
    
    triples = run_triplets_extraction(config, concept_data)
    
    # Step 3: Knowledge Graph Fusion
    fused_graph = run_graph_fusion(config, triples, relation_types, concept_data)
    
    # Step 4: Neo4j integration (commented out for now)
    # This will be implemented in a future version
    my_custom_logger.info("Neo4j integration is currently disabled.")
    
    my_custom_logger.info("GraphFusion Pipeline completed successfully.")
    return 0

if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)







