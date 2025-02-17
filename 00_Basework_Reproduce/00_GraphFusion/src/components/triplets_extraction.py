import json
from langchain_core.prompts import ChatPromptTemplate
from collections import Counter
from tqdm import tqdm

def extract_candidate_triples(
    language_model: any,
    output_file_path: str,
    relation_definitions: dict[str, dict[str, str]],
    concept_data: dict[str, dict[str, list[str]]],
    logger: any,
    parameters: dict[str, any]
) -> None:
    """
    Candidate Triple Extraction

    Extracts candidate triples from the given 'concept_data' (which maps concept names
    to a list of abstracts). Writes each extracted triple as JSON lines to 'output_file_path'.

    :param language_model: The language model to use (must have a .invoke(prompt) method).
    :param output_file_path: File path to write the extracted triples (JSON lines).
    :param relation_definitions: Definitions of possible relations {rel_type: {"description": "..."}}
    :param concept_data: Data mapping concept_name -> {"abstracts": [...], ...}
    :param logger: A logger object for logging messages (info, debug, etc.).
    :param parameters: A configuration dictionary. Recognized keys:
                      - "prompt_template_file": path to the triple-extraction prompt template
                      - "max_chars": maximum number of characters from the abstracts to process at a time
    :return: None
    """

    # Ensure defaults if not provided
    if "prompt_template_file" not in parameters:
        parameters["prompt_template_file"] = "prompts/prompt_tpextraction.txt"
        logger.info(f"No prompt template specified. Using default: {parameters['prompt_template_file']}")

    if "max_chars" not in parameters['split']:
        parameters['split']["max_chars"] = 10000
        logger.info(f"No max_chars specified. Using default: {parameters['split']['max_chars']}")

    logger.info("Starting candidate triple extraction...")

    # Load prompt template from file
    with open(parameters["prompt_template_file"], "r", encoding="utf-8") as pt_file:
        prompt_template_str = pt_file.read()

    # Initialize ChatPromptTemplate
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", "You are a knowledge graph builder."),
        ("user", prompt_template_str)
    ])

    # Open output stream
    with open(output_file_path, "w", encoding="utf-8") as out_file:
        extracted_relations = []

        # Enumerate over the concept_data dictionary
        for concept_index, (concept_key, concept_info) in tqdm(
            enumerate(concept_data.items()), 
            total=len(concept_data),
            desc="Extracting triples"
        ):
            # Combine abstracts into a single string (truncate if needed)
            joined_abstracts = " -- ".join(concept_info["abstracts"])[:parameters['split']['max_chars']]

            # Build the prompt with the needed variables
            prompt = prompt_template.invoke({
                "abstract": joined_abstracts,
                "concept": [concept_key],
                "relation_definitions": "\n".join(
                    f"{r_type}: {r_data['description']}" 
                    for r_type, r_data in relation_definitions.items()
                )
            })

            # Call the language model
            model_output = language_model.invoke(prompt)

            # Parse output if it's not "None" (assuming valid JSON returned)
            if model_output != "None":
                try:
                    parsed_output = json.loads(model_output)
                except json.JSONDecodeError:
                    logger.warning(f"JSON decoding failed for concept '{concept_key}'. Skipping.")
                    continue

                for triple_entry in parsed_output:
                    triple_relation = triple_entry.get("p", None)
                    if not triple_relation or triple_relation not in relation_definitions:
                        continue

                    # Collect the relation type
                    extracted_relations.append(triple_relation)

                    # Add context to the triple, then write to file as JSON
                    triple_entry["concept_index"] = concept_index
                    triple_entry["concept_name"] = concept_key
                    out_file.write(json.dumps(triple_entry) + "\n")

    logger.info("Candidate triple extraction completed.")
    logger.info(f"Number of extracted candidate triples: {len(extracted_relations)}")
    logger.debug(f"Extracted triples by relation type: {Counter(extracted_relations)}")
