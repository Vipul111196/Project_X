## 1. Setup & Imports
import os
from dotenv import load_dotenv
import yaml
from neo4j_graphrag.indexes import create_vector_index
from src.components.utils import process_pdfs_in_directory, getSchemaFromOnto
import pickle  
from src.components.models import KnowledgeGraphLLM
from langchain_openai import OpenAIEmbeddings
from src.pipelines.custom_KG import CustomKGPipeline
from src.components.document_processor import DocumentPreprocessor
from src.pipelines.raptor import TextClusterSummarizer

# from neo4j_graphrag.experimental.components.resolver import SinglePropertyExactMatchResolver
import asyncio
from rdflib import Graph as RDFGraph
from neo4j import GraphDatabase

# 
import os
from dotenv import load_dotenv
## Load environment variables from .env file
load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
uri = os.getenv("NEO4J_URI")
user = os.getenv("NEO4J_USERNAME")
password = os.getenv("NEO4J_PASSWORD")
AUTH = (user, password)

# Connect to Neo4j
driver = GraphDatabase.driver(uri, auth=AUTH)

# Loading the config file
with open("config/config.yaml", "r") as f:
    config = yaml.safe_load(f)

# Create Vector Index
if config['Neo4j_Database']['create'] == 1:
    INDEX_NAME = config['Neo4j_Database']['index_name']
    DIMENSION = config['Neo4j_Database']['dimensions']
    SIMILIARITY_FN = config['Neo4j_Database']['similarity_fn']
    create_vector_index(
        driver,
        INDEX_NAME,
        label="Chunk",
        embedding_property="embedding",
        dimensions=DIMENSION,
        similarity_fn=SIMILIARITY_FN,
)
    
# Reading the schema from the ontology file
if config['Knowledge_graph']['ontology_schema_present'] == 1:
    onto_file_path = config['Knowledge_graph']['ontology_schema_path']
    g = RDFGraph()
    g.parse(onto_file_path, format="turtle")
    schema_model = getSchemaFromOnto(g)

    print(f"Schema model loaded from {onto_file_path}")
    # Saving the schema model to a file
    with open(config['Knowledge_graph']['extracted_schema_path'], 'w') as file:

        classes = set([e["label"] for e in schema_model.entities.values()])
        obj_props = set([r["label"] for r in schema_model.relations.values()])
        data_props = set([p["name"] for e in schema_model.entities.values() for p in e.get("properties", [])])
        file.write(f"Entities:  {classes}\n")
        file.write(f"Relations: {obj_props}\n")
        file.write(f"Properties: {data_props}\n")
        file.write(f"Schema Model: {schema_model.potential_schema}")    

    print(f"Schema model extracted as text and saved")

# # Loading the model 
LLM_model_name = config['model']['LLM_model_name']
embedding_model_name = config['model']['embedding_model_name']
llm = KnowledgeGraphLLM(model_name=LLM_model_name, max_tokens=10000, api_key=api_key)
embedding_model = OpenAIEmbeddings(api_key=api_key, model=embedding_model_name)

# Instantiate and run
preprocessor = DocumentPreprocessor(config=config, pdf_processor=process_pdfs_in_directory)
chunks = preprocessor.run()

summarizer = TextClusterSummarizer(token_limit=17000, chunks=chunks, model_name="gpt-4o-mini", embedding_model_name="text-embedding-3-large")
# Run the summarizer and get the final output
final_output = summarizer.run()
with open(config['data_processing']['RAPTOR']['processed_path'], 'wb') as file:
    pickle.dump(final_output['documents'], file)

with open(config['data_processing']['RAPTOR']['processed_path'], 'rb') as file:
    raptor_chunks = pickle.load(file)


# Run the KG pipeline
pipeline = CustomKGPipeline(
driver=driver,
embedder=embedding_model,
kg_llm=llm,
classes=classes,
object_properties=obj_props,
data_properties=data_props,
prompt_template_path="prompt/prompt_2.txt",
)
pipeline.run(raptor_chunks)


print("Knowledge graph construction completed.")