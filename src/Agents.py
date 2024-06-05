## Initializes the agent used

from llama_index.llms.ollama import Ollama
from llama_index.core.agent import ReActAgent

from tools import CustomTools
from prompts import context

class Agents():
    def __init__(self):
        pass
    
    def make_agents():

        tools_used = CustomTools()
        tools = tools_used.make_tools()

        code_llm = Ollama(model="codellama")
        agent = ReActAgent.from_tools(tools, llm=code_llm, verbose=True, context=context)
        agents = [agent]
        return agents

