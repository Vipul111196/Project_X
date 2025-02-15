from typing_extensions import TypedDict, Annotated, List
from langchain_openai import ChatOpenAI
# from langchain_ollama import ChatOllama
from pydantic import BaseModel
from langchain_core.callbacks import CallbackManagerForLLMRun
from typing import Optional, Any
from langchain_core.language_models.llms import LLM
import json
import os
from dotenv import load_dotenv

# Load environment variables from .env file
# This handles the loading of API keys and other configuration
load_dotenv(".env")  # Use relative path instead of hardcoded path

class Triple(TypedDict):
    """
    Type definition for a knowledge graph triple.
    
    Attributes:
        s: Subject of the triple
        p: Predicate/relation of the triple
        o: Object of the triple
    """
    s: Annotated[str, ..., "Subject of the extracted Knowledge Graph Triple"]
    p: Annotated[str, ..., "Relation of the extracted Knowledge Graph Triple"]
    o: Annotated[str, ..., "Object of the extracted Knowledge Graph Triple"]


class Triples(BaseModel):
    """
    Pydantic model for a list of triples.
    Used for structured output from language models.
    """
    triples: List[Triple]


class KnowledgeGraphLLM(LLM):
    """
    Custom LLM wrapper specifically for knowledge graph extraction.
    
    This class wraps around OpenAI's ChatCompletion API and processes
    the output as structured triples for knowledge graph construction.
    
    Attributes:
        model_name: Name of the model to use (e.g., "gpt-4o-mini")
        max_tokens: Maximum tokens in the model response
        api_key: OpenAI API key
    """
    model_name: str
    max_tokens: int
    api_key: Optional[str] = None

    @property
    def _llm_type(self) -> str:
        """
        Return the type of LLM.
        """
        return f"Candidate Triple Extraction Chain based on {self.model_name}"

    def _call(
            self,
            prompt: str,
            stop: Optional[List[str]] = None,
            run_manager: Optional[CallbackManagerForLLMRun] = None,
            **kwargs: Any,
    ) -> str:
        """
        Call the language model with the given prompt and return structured triples.
        
        Args:
            prompt: The prompt to send to the language model
            stop: Optional stop sequences
            run_manager: Optional callback manager
            **kwargs: Additional keyword arguments
            
        Returns:
            str: JSON string of triples or "None" if no triples are returned
        """
        try:
            # If no API key was provided during initialization, use environment variable
            if not self.api_key:
                self.api_key = os.getenv("OPENAI_API_KEY")
                if not self.api_key:
                    raise ValueError("No API key provided and OPENAI_API_KEY not found in environment")
            
            # Initialize the model with structured output
            model = ChatOpenAI(
                model=self.model_name, 
                max_tokens=self.max_tokens, 
                api_key=self.api_key
            )
            # Alternatively use Ollama for local models
            # model = ChatOllama(model=self.model_name, base_url="http://localhost:11434/")
            
            # Configure the model to return structured output
            model = model.with_structured_output(Triples)
            
            # Stream the response and get the final result
            stream_response = model.stream(prompt)
            response = self.get_last_chunk(stream_response)

            # Check if the response contains triples
            if not hasattr(response, 'triples'):
                return "None"
                
            # Convert the triples to a JSON string and remove newlines
            return json.dumps(response.triples).replace('\n', '')
        except Exception as e:
            # Log error and return None on failure
            print(f"Error calling language model: {e}")
            return "None"

    @staticmethod
    def get_last_chunk(stream_response):
        """
        Extract the last chunk from a streamed response.
        
        Args:
            stream_response: Streamed response from the language model
            
        Returns:
            The last chunk of the response
        """
        last_chunk = None
        for chunk in stream_response:
            last_chunk = chunk
        return last_chunk