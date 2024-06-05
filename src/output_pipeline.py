from llama_index.llms.ollama import Ollama
from pydantic import BaseModel
from llama_index.core.output_parsers import PydanticOutputParser
from llama_index.core.query_pipeline import QueryPipeline
from llama_index.core import PromptTemplate

from src.prompts import code_parser_template


llm = Ollama(model="llama3", request_timeout=30.0)


class CodeOutput(BaseModel):
    code: str
    description: str
    filename: str

class Output_pipeline():
    def __init__(self):
        pass
    
    def give_output_pipeline():
        parser = PydanticOutputParser(CodeOutput)
        json_prompt_str = parser.format(code_parser_template)
        json_prompt_tmpl = PromptTemplate(json_prompt_str)
        output_pipeline = QueryPipeline(chain=[json_prompt_tmpl, llm])
        return output_pipeline

