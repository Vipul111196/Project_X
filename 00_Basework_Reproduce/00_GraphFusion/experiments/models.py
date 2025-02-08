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

load_dotenv("I:/11_DFKI_Hiwi/Work/01_Code/Graphusion/.env")

api_key = os.getenv("OPENAI_API_KEY")

class Triple(TypedDict):
    s: Annotated[str, ..., "Subject of the extracted Knowledge Graph Triple"]
    p: Annotated[str, ..., "Relation of the extracted Knowledge Graph Triple"]
    o: Annotated[str, ..., "Object of the extracted Knowledge Graph Triple"]


class Triples(BaseModel):
    triples: List[Triple]


class KnowledgeGraphLLM(LLM):
    model_name: str
    max_tokens: int
    api_key: str

    @property
    def _llm_type(self) -> str:
        return f"Candidate Triple Extraction Chain based on {self.model_name}"

    def _call(
            self,
            prompt: str,
            stop: Optional[List[str]] = None,
            run_manager: Optional[CallbackManagerForLLMRun] = None,
            **kwargs: Any,
    ) -> str:
        model = ChatOpenAI(model=self.model_name, max_tokens=self.max_tokens, api_key=self.api_key)
        # model = ChatOllama(model = self.model_name,base_url ="http://localhost:11434/")
        model = model.with_structured_output(Triples)
        stream_response = model.stream(prompt)
        response = self.get_last_chunk(stream_response)

        # if the object does not habe the attribute triples
        if not hasattr(response, 'triples'):
            return "None"
        return json.dumps(response.triples).replace('\n', '')

    @staticmethod
    def get_last_chunk(stream_response):
        last_chunk = None
        for chunk in stream_response:
            last_chunk = chunk
        return last_chunk