# Import necessary libraries and modules
import os
import nltk
import json
from tqdm import tqdm
import pandas as pd
from umap import UMAP
from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer
from bertopic.vectorizers import ClassTfidfTransformer
from bertopic.representation import KeyBERTInspired
from sentence_transformers import SentenceTransformer
from nltk.data import find
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from fuzzywuzzy import fuzz
from src.logger import logging

# Download NLTK packages if needed
def ensure_nltk_package(package_name: str):
    """
    Check if an NLTK package is downloaded; if not, download it.
    
    Args:
        package_name (str): Name of the NLTK package to check/download.
    """
    try:
        find(f"corpora/{package_name}")
    except LookupError:
        nltk.download(package_name)

# Lemmatize each word in a phrase to its singular form
def lemmatize_phrase(phrase: str) -> str:
    """
    Lemmatize each word in a phrase to its singular form.
    
    Args:
        phrase (str): The input phrase to lemmatize.
        
    Returns:
        str: The lemmatized phrase.
    """
    lemmatizer = WordNetLemmatizer()
    words = phrase.split()
    return " ".join(lemmatizer.lemmatize(word, wordnet.NOUN) for word in words)

# Filter documents based on fuzzy matching
def fuzzy_filter(term: str, documents: list[str], threshold: int = 70) -> list[str]:
    """
    Return all documents where the partial ratio to 'term' is >= threshold.
    
    Args:
        term (str): The term to match against documents.
        documents (list[str]): List of documents to filter.
        threshold (int, optional): Minimum fuzzy match score (0-100). Defaults to 70.
        
    Returns:
        list[str]: Filtered documents that match the term above the threshold.
    """
    filtered_docs = []
    term_lower = term.lower()
    for doc in documents:
        if isinstance(doc, str) and fuzz.partial_ratio(term_lower, doc.lower()) >= threshold:
            filtered_docs.append(doc)
    return filtered_docs

# Extract concepts and abstracts
def extract_concepts_and_abstracts(
    documents: list[str],
    output_concepts_file: str,
    output_concept_abstracts_json: str,
    logger,
    excluded_words: list[str] = None,
    parameters: dict = None
) -> None:
    """
    Extract candidate concepts using BERTopic from the given documents.
    Write the list of unique concepts to 'output_concepts_file' and
    a JSON file containing abstracts for each concept to 'output_concept_abstracts_json'.
    Also logs abstracts that were filtered out to a separate folder.

    Args:
        documents (list[str]): List of documents (strings) from which to extract concepts.
        output_concepts_file (str): File path to write extracted concepts (TXT format).
        output_concept_abstracts_json (str): File path to write the abstract(s) for each concept (JSON format).
        logger: Logger object for logging messages.
        excluded_words (list[str], optional): List of stop words to exclude from concept extraction. Defaults to None.
        parameters (dict, optional): Configuration dictionary. Recognized keys:
                          - "language" (default "english")
                          - "gold_concept_file" (path to gold concept file, optional)
                          - "filtered_abstracts_dir" (directory to log filtered abstracts, optional)
    """
    # Ensure default parameters
    if parameters is None:
        parameters = {}
    language = parameters.get("language", "english")
    gold_file = parameters.get("gold_concept_file", "")
    filtered_abstracts_dir = parameters.get("filtered_abstracts_dir", "outputs/filtered_abstracts")
    
    # Create the filtered abstracts directory if it doesn't exist
    os.makedirs(filtered_abstracts_dir, exist_ok=True)

    # Download NLTK packages if needed
    ensure_nltk_package("wordnet")
    ensure_nltk_package("omw-1.4")

    logger.info("Starting concept extraction...")

    # Set up models for BERTopic
    if language == "english":
        count_vectorizer = CountVectorizer(
            ngram_range=(2, 4),
            stop_words="english" if excluded_words is None else excluded_words
        )
        sentence_model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
    else:
        logger.warning(f"Language '{language}' is not directly supported. Defaulting to English models.")
        count_vectorizer = CountVectorizer(
            ngram_range=(2, 4),
            stop_words="english" if excluded_words is None else excluded_words
        )
        sentence_model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

    # Configure UMAP for dimensionality reduction
    umap_model = UMAP(
        n_neighbors=20,
        n_components=50,
        metric="cosine",
        min_dist=0.0,
        random_state=37
    )
    
    # Configure Class-based TF-IDF to extract important words per topic
    ctfidf = ClassTfidfTransformer(reduce_frequent_words=False)
    
    # Configure representation model
    representation = KeyBERTInspired()

    # Set up BERTopic model with all components
    topic_model = BERTopic(
        verbose=True,
        umap_model=umap_model,
        ctfidf_model=ctfidf,
        vectorizer_model=count_vectorizer,
        embedding_model=sentence_model,
        representation_model=representation,
        nr_topics=50,  # Maximum number of topics to detect
        low_memory=True,
        calculate_probabilities=False
    )

    # Fit BERTopic model to documents and transform them to get topic assignments
    topics, _ = topic_model.fit_transform(documents)
    all_topics = topic_model.get_topics()

    # Collect unique concepts from all topic keywords
    raw_concepts = []
    for topic_num, keywords in all_topics.items():
        if topic_num != -1:  # -1 is a catch-all topic for outliers
            raw_concepts.extend([word for word, _ in keywords])

    # Convert to lowercase and remove duplicates
    unique_concepts = list(set(concept.lower() for concept in raw_concepts))

    # Write extracted concepts to file
    with open(output_concepts_file, "w", encoding="utf-8") as file:
        for idx, concept in enumerate(unique_concepts, 1):
            file.write(f"{idx}|{concept}\n")
    logger.info(f"Extracted concepts written to {output_concepts_file}.")

    # Lemmatize/singularize concepts
    lemmatized_concepts = [lemmatize_phrase(c) for c in unique_concepts]
    df_extracted_concepts = pd.DataFrame(lemmatized_concepts, columns=["concept"])
    df_extracted_concepts["label"] = 0

    # If a gold concept file is provided and found, merge gold concepts with extracted ones
    if gold_file and os.path.exists(gold_file):
        gold_data = pd.read_csv(gold_file, delimiter="|", header=None)
        # Column 1 in gold_data is the concept (assuming ID|concept format)
        gold_concept_list = gold_data[1].tolist()

        # Lemmatize gold concepts, convert to lowercase, and add label 1
        processed_gold_concepts = [lemmatize_phrase(gc.lower()) for gc in gold_concept_list]
        df_gold = pd.DataFrame(processed_gold_concepts, columns=["concept"])
        df_gold["label"] = 1

        df_extracted_concepts = pd.concat([df_extracted_concepts, df_gold], ignore_index=True)
        df_extracted_concepts.sort_values(by="label", inplace=True)
    elif gold_file:
        logger.warning(f"Gold concept file '{gold_file}' was not found. Skipping gold concept merge.")

    # Remove duplicates after merging
    df_extracted_concepts.drop_duplicates(subset="concept", keep="first", inplace=True)

    # Create a dictionary to track filtered abstracts
    filtered_abstracts = {}

    # For each concept, collect documents where the concept appears (fuzzy matching)
    concept_to_abstracts = {}
    for _, row in tqdm(df_extracted_concepts.iterrows(), desc="Gathering abstracts", total=len(df_extracted_concepts)):
        concept_text = row["concept"]
        concept_label = row["label"]
        matched_abstracts = fuzzy_filter(concept_text, documents)
        
        # Track abstracts that match this concept
        concept_to_abstracts[concept_text] = {
            "abstracts": matched_abstracts,
            "label": concept_label
        }
        
        # Log which abstracts are filtered for this concept
        for doc in matched_abstracts:
            if doc in filtered_abstracts:
                filtered_abstracts[doc].append(concept_text)
            else:
                filtered_abstracts[doc] = [concept_text]

    # Write the dictionary of concept abstracts to JSON
    with open(output_concept_abstracts_json, "w", encoding="utf-8") as file:
        json.dump(concept_to_abstracts, file, ensure_ascii=False, indent=4)
    logger.info(f"Abstracts for concepts written to {output_concept_abstracts_json}.")
    
    # Save filtered abstracts information
    # Identify abstracts that weren't matched to any concept
    unmatched_abstracts = [doc for doc in documents if doc not in filtered_abstracts]
    
    # Save unmatched abstracts
    with open(os.path.join(filtered_abstracts_dir, "unmatched_abstracts.txt"), "w", encoding="utf-8") as file:
        for doc in unmatched_abstracts:
            file.write(f"{doc}\n\n---\n\n")
    logger.info(f"Saved {len(unmatched_abstracts)} unmatched abstracts to {os.path.join(filtered_abstracts_dir, 'unmatched_abstracts.txt')}")
    
    # Save matched abstracts with their associated concepts
    with open(os.path.join(filtered_abstracts_dir, "matched_abstracts.json"), "w", encoding="utf-8") as file:
        json.dump(filtered_abstracts, file, ensure_ascii=False, indent=4)
    logger.info(f"Saved matched abstracts with their concepts to {os.path.join(filtered_abstracts_dir, 'matched_abstracts.json')}")

    # Logging basic stats
    logger.info("Concept extraction completed.")
    logger.info(f"Total unique concepts from BERTopic: {len(lemmatized_concepts)}")
    if gold_file and os.path.exists(gold_file):
        new_concepts_count = sum(1 for info in concept_to_abstracts.values() if info["label"] == 0)
        logger.info(f"Number of newly added concepts (label=0): {new_concepts_count}")

    empty_abstracts = sum(1 for info in concept_to_abstracts.values() if not info["abstracts"])
    logger.info(f"Number of concepts with no matched abstracts: {empty_abstracts}")
    logger.info(f"Total abstracts processed: {len(documents)}")
    logger.info(f"Abstracts matched to at least one concept: {len(filtered_abstracts)}")
    logger.info(f"Abstracts not matched to any concept: {len(unmatched_abstracts)}")
