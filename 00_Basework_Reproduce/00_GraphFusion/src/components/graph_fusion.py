# import json
# import logging
# import random
# import os
# import pandas as pd
# from langchain_core.prompts import ChatPromptTemplate
# from tqdm import tqdm

# def fuse_triples(model, candidate_triples, relation_def, data):
    
#     fused_triples = []
#     # For each unique concept (from candidate triples), create a textual summary of its related triples
#     candidate_concepts = set(data.keys())

#     for concept in tqdm(list(candidate_concepts)):
#         # Create a simple string representation (list all triples involving this concept)
#         candidate_subgraph = []
#         for triple in candidate_triples:
#             if concept in triple:
#                 candidate_subgraph.append(f"({triple[0]}, {triple[1]}, {triple[2]})")
                
#         # Get background abstracts (if available)
#         background = ""
#         if concept in data:
#             background = ' --  '.join(data[concept]['abstracts'])
#         else:
#             print(f"No background found for concept {concept}")
#             background = "No background information available."
        
#         # Build the fusion prompt using a simple formatted string
#         prompt_template_txt = open("prompts/prompt_fusion.txt").read()
#         # print(prompt_template_txt)
#         prompt_template = ChatPromptTemplate.from_template(prompt_template_txt)
#         # print('Prompt template', prompt_template)   
        
#         prompt = prompt_template.invoke({
#             "concept": concept,
#             "graph1": candidate_subgraph,
#             "graph2": fused_triples,
#             "background": background,
#             "relation_definitions": '\n'.join([f"{k}: {v['description']}" for k, v in relation_def.items()])
#         })

#         response = model.invoke(prompt)
#         print(f"Response for concept {concept}: {response}")    

#         print("/n /n /n") 

#         if response and response != "None":
#             try:
#                 response_json = json.loads(response)
#                 for triple in response_json:
#                     # Validate the relation exists in your definitions
#                     if triple['p'] in relation_def:
#                         fused_triples.append((triple['s'], triple['p'], triple['o']))
#             except Exception as e:
#                 print(f"Error fusing triples for concept {concept}: {e}")
        
#         fused_triples = list(set(fused_triples))
#         print(f"Total fused triples: {len(fused_triples)}")
        
#     return fused_triples


# ####
# # logging pending

# Note: Makes new changes in the code snippet below experimented in Jupyter Notebook for fusion of triplets

# kg_fusion.py
import json
import logging
import os
from langchain_core.prompts import ChatPromptTemplate
from tqdm import tqdm

def perform_knowledge_fusion(
    language_model: any,
    candidate_triples: list[tuple[str, str, str]],
    relation_definitions: dict[str, dict[str, str]],
    concept_data: dict[str, dict[str, list[str]]],
    logger: logging.Logger,
    prompt_fusion_path: str,
    fused_triplets_path: str
) -> list[tuple[str, str, str]]:
    """
    Perform knowledge graph fusion by merging new triples into an evolving graph
    based on a language model's suggestions.

    :param language_model: A model object with an .invoke(prompt) method returning JSON or "None".
    :param candidate_triples: List of (subject, predicate, object) tuples to be fused.
    :param relation_definitions: Dictionary of { relation_type: {"description": "..."} }.
    :param concept_data: Maps concept_name -> {"abstracts": [...], ...} for background info.
    :param logger: Logger instance for logging messages.
    :param prompt_fusion_path: File path to the prompt template used for fusion.
    :return: A list of unique fused triples.
    """

    fused_triples = []
    # Convert concept_data keys to a set for iteration
    unique_concepts = set(concept_data.keys())

    # Iterate through each concept present in the data
    for concept in tqdm(list(unique_concepts), desc="Fusing concepts"):
        # Gather all candidate triples that mention the concept
        concept_subgraph_str = []
        for triple in candidate_triples:
            if concept in triple:
                concept_subgraph_str.append(f"({triple[0]}, {triple[1]}, {triple[2]})")

        # Retrieve background abstracts if present
        if concept in concept_data:
            background_text = " -- ".join(concept_data[concept].get("abstracts", []))
        else:
            background_text = "No background information available."
            logger.warning(f"No background found for concept '{concept}'")

        # Load the fusion prompt template
        if not os.path.isfile(prompt_fusion_path):
            logger.error(f"Fusion prompt file '{prompt_fusion_path}' not found.")
            return fused_triples  # Return what we have so far or raise an exception

        with open(prompt_fusion_path, "r", encoding="utf-8") as p_file:
            prompt_template_str = p_file.read()

        prompt_template = ChatPromptTemplate.from_template(prompt_template_str)

        # Build the final prompt
        prompt = prompt_template.invoke({
            "concept": concept,
            "graph1": concept_subgraph_str,   # The local subgraph relevant to this concept
            "graph2": fused_triples,         # The current global fused graph
            "background": background_text,
            "relation_definitions": "\n".join([
                f"{rel_type}: {desc['description']}"
                for rel_type, desc in relation_definitions.items()
            ])
        })

        # Invoke language model
        response = language_model.invoke(prompt)
        logger.info(f"Fusion response for concept '{concept}': {response}")

        if response and response.lower() != "none":
            try:
                # Attempt to parse JSON
                parsed_triples = json.loads(response)
                if isinstance(parsed_triples, list):
                    for triple_item in parsed_triples:
                        rel_type = triple_item.get("p")
                        if rel_type in relation_definitions:
                            # Collect (subject, predicate, object)
                            fused_triples.append((
                                triple_item.get("s", ""),
                                rel_type,
                                triple_item.get("o", "")
                            ))
                else:
                    logger.warning(f"Expected a list of triples in JSON for '{concept}' but got something else.")
            except json.JSONDecodeError as e:
                logger.error(f"JSON decoding failed for concept '{concept}': {e}")
        else:
            logger.debug(f"No valid response for concept '{concept}'.")

        # De-duplicate the fused triples
        fused_triples = list(set(fused_triples))
        logger.info(f"Current total fused triples after concept '{concept}': {len(fused_triples)}")

    # Save the fused triples to a file
    with open(fused_triplets_path, "w", encoding="utf-8") as f_file:
        json.dump(fused_triples, f_file, indent=2)

    return fused_triples
