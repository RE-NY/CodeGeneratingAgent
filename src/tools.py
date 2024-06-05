from llama_parse import LlamaParse
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, PromptTemplate
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.embeddings import resolve_embed_model
from llama_index.llms.ollama import Ollama

from llama_index.core.tools import FunctionTool
from src.prompts import code_reader_description
from src.utils import code_reader_func

from load_dotenv import load_dotenv
load_dotenv()



class CustomTools:
    def __init__(self):
        pass

    def make_tools():
        '''Returns list of tools that the agent can utilize as needed'''
        llm = Ollama(model="llama3", request_timeout=30.0)

        parser = LlamaParse(result_type="markdown")

        file_extractor = {".pdf": parser}
        documents = SimpleDirectoryReader("./data", file_extractor=file_extractor).load_data()

        embed_model = resolve_embed_model("local:BAAI/bge-m3")
        vector_index = VectorStoreIndex.from_documents(documents, embed_model=embed_model)
        query_engine = vector_index.as_query_engine(llm=llm)
        ##query engine for 1st tool is created

        code_reader = FunctionTool.from_defaults(
            fn=code_reader_func,
            name="code_reader",
            description=code_reader_description,
        )
        ##2nd tool is created


        tools = [
                QueryEngineTool(
                    query_engine=query_engine,
                    metadata=ToolMetadata(
                    name="api_documentation",
                    description="this gives documentation about code for an API. Use this for reading docs for the API",
                    ),
                ),
                code_reader,
        ]
        return tools 