# from typing_extensions import TypedDict, Annotated, List, Dict
# from langchain_openai import ChatOpenAI
# # from langchain_ollama import ChatOllama
# from pydantic import BaseModel, Field
# from langchain_core.callbacks import CallbackManagerForLLMRun
# from typing import Optional, Any
# from langchain_core.language_models.llms import LLM
# import json
# import os
# from dotenv import load_dotenv

# load_dotenv("I:/11_DFKI_Hiwi/Work/01_Code/Graphusion/.env")

# api_key = os.getenv("OPENAI_API_KEY")


# ########################################################
# #  1) Define a KeyValueAttribute model (no free dicts)  #
# ########################################################

# class KeyValueAttribute(BaseModel):
#     """Represents one attribute as a key-value pair."""
#     key: str = Field(..., description="Attribute name")
#     value: str = Field(..., description="Attribute value")

#     class Config:
#         extra = "forbid"
#         json_schema_extra = {
#             "type": "object",
#             "additionalProperties": False
#         }

# ########################################################
# #  2) Define your Triple + Triples with KV attributes  #
# ########################################################

# class Triple(BaseModel):
#     subject: str = Field(..., description="Subject text")
#     subject_type: str = Field(..., description="Class/type of the Subject")
#     relation: str = Field(..., description="Relation from subject to object")
#     object: str = Field(..., description="Object text")
#     object_type: str = Field(..., description="Class/type of the Object")

#     subject_attributes: List[KeyValueAttribute] = Field(
#         default_factory=list,
#         description="List of arbitrary attributes describing the Subject"
#     )
#     object_attributes: List[KeyValueAttribute] = Field(
#         default_factory=list,
#         description="List of arbitrary attributes describing the Object"
#     )

#     class Config:
#         extra = "forbid"
#         json_schema_extra = {
#             "type": "object",
#             "additionalProperties": False
#         }

# class Triples(BaseModel):
#     triples: List[Triple] = Field(..., description="All extracted knowledge graph triples")

#     class Config:
#         extra = "forbid"
#         json_schema_extra = {
#             "type": "object",
#             "additionalProperties": False
#         }


# class KnowledgeGraphLLM(LLM):
#     model_name: str
#     max_tokens: int
#     api_key: str

#     @property
#     def _llm_type(self) -> str:
#         return f"Candidate Triple Extraction Chain based on {self.model_name}"

#     def _call(
#             self,
#             prompt: str,
#             stop: Optional[List[str]] = None,
#             run_manager: Optional[CallbackManagerForLLMRun] = None,
#             **kwargs: Any,
#     ) -> str:
#         model = ChatOpenAI(model=self.model_name, max_tokens=self.max_tokens, api_key=self.api_key)
#         # model = ChatOllama(model = self.model_name,base_url ="http://localhost:11434/")
#         model = model.with_structured_output(Triples)
#         stream_response = model.stream(prompt)
#         response = self.get_last_chunk(stream_response)

#         # if the object does not habe the attribute triples
#         if not hasattr(response, 'triples'):
#             return "None"
#         return json.dumps(response.triples).replace('\n', '')

#     @staticmethod
#     def get_last_chunk(stream_response):
#         last_chunk = None
#         for chunk in stream_response:
#             last_chunk = chunk
#         return last_chunk

import json
from typing import List, Optional, Any
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.llms import LLM
from langchain_openai import ChatOpenAI

# Example Pydantic 2.x schema (Triples) that has no "additionalProperties" problems:
from typing import List
from pydantic import BaseModel, Field, ConfigDict

class KeyValueAttribute(BaseModel):
    """
    Represents one attribute as a key-value pair.
    NOTE: Pydantic 2.0 syntax with model_config
    """
    key: str = Field(..., description="Attribute name")
    value: str = Field(..., description="Attribute value")

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "type": "object",
            "additionalProperties": False
        }
    )

class Triple(BaseModel):
    subject: str = Field(..., description="Subject of the triple")
    subject_type: str = Field(..., description="Class/type of the Subject")
    relation: str = Field(..., description="Relation from subject to object")
    object: str = Field(..., description="Object of the triple")
    object_type: str = Field(..., description="Class/type of the Object")

    subject_attributes: List[KeyValueAttribute] = Field(
        default_factory=list,
        description="List of key-value attributes for the Subject"
    )
    object_attributes: List[KeyValueAttribute] = Field(
        default_factory=list,
        description="List of key-value attributes for the Object"
    )

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "type": "object",
            "additionalProperties": False
        }
    )

class Triples(BaseModel):
    """Top-level container for the list of Triple objects."""
    triples: List[Triple] = Field(..., description="All extracted knowledge graph triples")

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "type": "object",
            "additionalProperties": False
        }
    )


class KnowledgeGraphLLM(LLM):
    """
    Updated LLM class that uses OpenAI structured output for triple extraction.
    """
    model_name: str
    max_tokens: int
    api_key: str

    @property
    def _llm_type(self) -> str:
        # LLM's type name (for logging)
        return f"Candidate Triple Extraction Chain based on {self.model_name}"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """Invoke the LLM with a prompt and return the JSON string of extracted triples."""
        structured_model = ChatOpenAI(
            model=self.model_name,
            max_tokens=self.max_tokens,
            api_key=self.api_key
        ).with_structured_output(Triples)

        # Use .invoke(...) to get a typed Pydantic object
        response = structured_model.invoke(prompt)

        # If no 'triples' attribute is present, return "None"
        if not hasattr(response, "triples"):
            return "None"

        # Return as a JSON string (with optional pretty-print)
        # Pydantic 2.0: use .model_dump() and then json.dumps(...)
        as_dict = response.model_dump()
        return json.dumps(as_dict, ensure_ascii=False)
