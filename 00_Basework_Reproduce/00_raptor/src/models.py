"""
RAPTOR Model Classes

This module contains model implementations for the RAPTOR pipeline:
- QA models (answer generation)
- Summarization models (hierarchical tree node summarization)
- Embedding models (vector representation of text)

Models are implemented for both OpenAI and Hugging Face.
"""

import os
import sys
import yaml
import logging
import torch
import openai
from typing import Dict, Any, List, Union, Optional

from src.components import BaseSummarizationModel, BaseQAModel, BaseEmbeddingModel

# Import necessary libraries for Hugging Face models
try:
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
    from sentence_transformers import SentenceTransformer
    HUGGINGFACE_AVAILABLE = True
except ImportError:
    logging.warning("Hugging Face libraries not available. Only OpenAI models will work.")
    HUGGINGFACE_AVAILABLE = False

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

def load_environment_variables() -> None:
    """Load environment variables from .env file if available"""
    try:
        from dotenv import load_dotenv
        load_dotenv()  # Load from .env in current directory
                
    except ImportError:
        logging.warning("python-dotenv not installed. Using system environment variables.")

# Load environment variables
load_environment_variables()

# Load configuration
config = load_config()

# Log API keys status (first few chars only for security)
api_key = os.environ.get("OPENAI_API_KEY")
hf_token = os.environ.get(config["raptor"]["models"]["huggingface"]["token_env"])

logging.info(f"OpenAI API key: {'Present' if api_key else 'Not found'}")
logging.info(f"Hugging Face token: {'Present' if hf_token else 'Not found'}")

# ===== OpenAI Models ===== #

class OpenAIQAModel(BaseQAModel):
    """OpenAI-based question answering model"""
    
    def __init__(self, model_name: str = "gpt-4o-mini", api_key: str = api_key, **params):
        """
        Initialize the OpenAI QA model
        
        Args:
            model_name: Name of the OpenAI model to use
            api_key: OpenAI API key (optional, will use env var if not provided)
            params: Additional parameters for the OpenAI API
        """
        self.model_name = model_name
        self.params = params
        self.api_key = api_key
        
        # Initialize OpenAI client with API key if provided
        if self.api_key:
            self.client = openai.OpenAI(api_key=self.api_key)
        else:
            # Check if API key is in environment
            env_api_key = os.environ.get("OPENAI_API_KEY")
            if env_api_key:
                self.client = openai.OpenAI(api_key=env_api_key)
            else:
                logging.error("No OpenAI API key found in environment or passed to the model")
                self.client = None
                
        logging.info(f"Initialized OpenAI QA model: {model_name}")

    def answer_question(self, context: str, question: str) -> str:
        """
        Generate an answer to a question given the context
        
        Args:
            context: The context text to use for answering
            question: The question to answer
            
        Returns:
            str: The generated answer
        """
        # Check if client is initialized
        if not self.client:
            error_msg = "OpenAI client not initialized due to missing API key"
            logging.error(error_msg)
            return f"Error: {error_msg}"
            
        messages = [
            {"role": "system", "content": "You are an expert question-answering assistant."},
            {"role": "user", "content": f"Context: {context}\n\nQuestion: {question}\n\nAnswer concisely and completely."}
        ]
        
        # Use default params with any provided params
        params = {
            "max_tokens": 256,
            "temperature": 0.7,
            "top_p": 0.95
        }
        params.update(self.params)
        
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                **params
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logging.error(f"Error calling OpenAI API: {e}")
            return f"Error generating answer: {str(e)}"


class OpenAISummarizationModel(BaseSummarizationModel):
    """OpenAI-based summarization model"""
    
    def __init__(self, model_name: str = "gpt-4o-mini", api_key: str = api_key, **params):
        """
        Initialize the OpenAI summarization model
        
        Args:
            model_name: Name of the OpenAI model to use
            api_key: OpenAI API key (optional, will use env var if not provided)
            params: Additional parameters for the OpenAI API
        """
        self.model_name = model_name
        self.params = params
        self.api_key = api_key
        
        # Initialize OpenAI client with API key if provided
        if self.api_key:
            self.client = openai.OpenAI(api_key=self.api_key)
        else:
            # Check if API key is in environment
            env_api_key = os.environ.get("OPENAI_API_KEY")
            if env_api_key:
                self.client = openai.OpenAI(api_key=env_api_key)
            else:
                logging.error("No OpenAI API key found in environment or passed to the model")
                self.client = None
                
        logging.info(f"Initialized OpenAI summarization model: {model_name}")

    def summarize(self, context: str, max_tokens: int = 150) -> str:
        """
        Generate a summary of the context text
        
        Args:
            context: The text to summarize
            max_tokens: Maximum length of the summary
            
        Returns:
            str: The generated summary
        """
        # Check if client is initialized
        if not self.client:
            error_msg = "OpenAI client not initialized due to missing API key"
            logging.error(error_msg)
            return f"Error: {error_msg}"
            
        messages = [
            {"role": "system", "content": "You are an expert summarization assistant."},
            {"role": "user", "content": f"Please summarize the following text concisely:\n\n{context}"}
        ]
        
        # Use default params with any provided params and override max_tokens
        params = {
            "max_tokens": max_tokens,
            "temperature": 0.3,
            "top_p": 0.95
        }
        params.update(self.params)
        
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                **params
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logging.error(f"Error calling OpenAI API: {e}")
            return f"Error generating summary: {str(e)}"


# ===== Hugging Face Models ===== #

class HFUnifiedModel(BaseSummarizationModel, BaseQAModel):
    """Unified Hugging Face model for both QA and summarization to save memory"""
    
    def __init__(self, model_name: str = "meta-llama/Llama-3.1-8B-Instruct", 
                 token: Optional[str] = hf_token, 
                 device: str = "auto",
                 quantization: str = "fp16",
                 qa_params: Dict[str, Any] = None,
                 summarization_params: Dict[str, Any] = None):
        """
        Initialize the unified Hugging Face model
        
        Args:
            model_name: Name of the Hugging Face model to use
            token: Hugging Face token for accessing gated models
            device: Device to run the model on ("auto", "cpu", "cuda", or "mps")
            quantization: Quantization strategy ("none", "fp16", "int8", or "int4")
            qa_params: Parameters for question answering
            summarization_params: Parameters for summarization
        """
        if not HUGGINGFACE_AVAILABLE:
            raise ImportError("Hugging Face libraries are not available. Please install them.")
            
        self.model_name = model_name
        self.qa_params = qa_params or {}
        self.summarization_params = summarization_params or {}
        
        # Determine the device to use
        if device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        self.device = device
        
        # Set up quantization
        torch_dtype = torch.float32  # Default
        if quantization == "fp16":
            torch_dtype = torch.float16
        
        # Load tokenizer and model
        logging.info(f"Loading {model_name} on {device} with {quantization} quantization...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, token=token)
        
        # Load model with appropriate quantization
        load_args = {
            "token": token,
            "torch_dtype": torch_dtype,
            "low_cpu_mem_usage": True,
        }
        
        # Add int8 or int4 quantization if specified
        if quantization == "int8":
            load_args["load_in_8bit"] = True
        elif quantization == "int4":
            load_args["load_in_4bit"] = True
            
        try:
            self.model = AutoModelForCausalLM.from_pretrained(model_name, **load_args)
            
            # Move model to device if not using quantization that already handles it
            if quantization not in ["int8", "int4"] and device != "cpu":
                self.model = self.model.to(device)
                
            # Create text generation pipeline
            self.pipeline = pipeline(
                "text-generation", 
                model=self.model, 
                tokenizer=self.tokenizer,
                device=0 if device == "cuda" else device  # device mapping for pipeline
            )
            
            logging.info(f"Successfully initialized {model_name} model")
        except Exception as e:
            logging.error(f"Error loading model {model_name}: {e}")
            raise

    def answer_question(self, context: str, question: str) -> str:
        """
        Generate an answer to a question given the context
        
        Args:
            context: The context text to use for answering
            question: The question to answer
            
        Returns:
            str: The generated answer
        """
        # Format for Llama-style models (works for most instruction models)
        prompt = f"""<|system|>
You are a helpful assistant that answers questions based on the provided context.
</s>
<|user|>
Context: {context}

Question: {question}
</s>
<|assistant|>"""
        
        # Default parameters
        params = {
            "max_new_tokens": 256,
            "do_sample": True,
            "temperature": 0.7
        }
        # Override with provided parameters
        params.update(self.qa_params)
        
        try:
            output = self.pipeline(prompt, **params)
            
            # Extract just the assistant's response
            result = output[0]['generated_text']
            # Remove the prompt from the result
            response = result.split("<|assistant|>")[-1].strip()
            return response
        except Exception as e:
            logging.error(f"Error generating answer with {self.model_name}: {e}")
            return f"Error generating answer: {str(e)}"

    def summarize(self, context: str, max_tokens: int = 150) -> str:
        """
        Generate a summary of the context text
        
        Args:
            context: The text to summarize
            max_tokens: Maximum length of the summary
            
        Returns:
            str: The generated summary
        """
        # Format for Llama-style models (works for most instruction models)
        prompt = f"""<|system|>
You are a helpful assistant that summarizes text accurately and concisely.
</s>
<|user|>
Please summarize the following text:

{context}
</s>
<|assistant|>"""
        
        # Default parameters
        params = {
            "max_new_tokens": max_tokens,
            "do_sample": False,
            "temperature": 0.3
        }
        # Override with provided parameters
        params.update(self.summarization_params)
        
        try:
            output = self.pipeline(prompt, **params)
            
            # Extract just the assistant's response
            result = output[0]['generated_text']
            # Remove the prompt from the result
            response = result.split("<|assistant|>")[-1].strip()
            return response
        except Exception as e:
            logging.error(f"Error generating summary with {self.model_name}: {e}")
            return f"Error generating summary: {str(e)}"


# ===== Embedding Models ===== #

class OpenAIEmbeddingModel(BaseEmbeddingModel):
    
    def __init__(self, model_name: str = "text-embedding-3-small", api_key: str = api_key, **params):
        """
        Initialize the OpenAI embedding model
        
        Args:
            model_name: Name of the OpenAI embedding model to use
            api_key: OpenAI API key (optional, will use env var if not provided)
            params: Additional parameters for the OpenAI API
                - dimensions: Output dimension for the embedding
                - encoding_format: 'float' or 'base64'
        """
        self.model_name = model_name
        self.api_key = api_key
        self.params = params
        
        # Initialize OpenAI client
        if self.api_key:
            self.client = openai.OpenAI(api_key=self.api_key)
        else:
            # Check if API key is in environment
            env_api_key = os.environ.get("OPENAI_API_KEY")
            if env_api_key:
                self.client = openai.OpenAI(api_key=env_api_key)
            else:
                logging.error("No OpenAI API key found in environment or passed to the model")
                self.client = None

    def create_embedding(self, text: str) -> List[float]:
        """Create embedding for the given text using OpenAI API"""
        try:
            response = self.client.embeddings.create(
                model=self.model_name,
                input=text,
                **self.params
            )
            return response.data[0].embedding
        except Exception as e:
            logging.error(f"Error creating OpenAI embedding: {e}")
            return []


class SentenceTransformerEmbedding(BaseEmbeddingModel):
    """Sentence Transformer embedding model"""
    
    def __init__(self, model_name: str = "sentence-transformers/multi-qa-mpnet-base-cos-v1", 
                 device: str = "auto"):
        """
        Initialize the Sentence Transformer embedding model
        
        Args:
            model_name: Name of the model to use
            device: Device to run the model on ("auto", "cpu", "cuda", or "mps")
        """
        if not HUGGINGFACE_AVAILABLE:
            raise ImportError("Sentence Transformers library is not available. Please install it.")
            
        # Determine the device to use
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
            
        self.model_name = model_name
        try:
            self.model = SentenceTransformer(model_name, device=device)
            logging.info(f"Initialized {model_name} embedding model on {device}")
        except Exception as e:
            logging.error(f"Error loading embedding model {model_name}: {e}")
            raise

    def create_embedding(self, text: str) -> List[float]:
        """
        Create embedding for the given text
        
        Args:
            text: The text to embed
            
        Returns:
            List[float]: The embedding vector
        """
        try:
            return self.model.encode(text)
        except Exception as e:
            logging.error(f"Error creating embedding: {e}")
            raise


