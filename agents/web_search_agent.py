from datetime import datetime
from pydantic import BaseModel
from mirascope.core import (
    BaseMessageParam,
    Messages,
    openai,
    prompt_template,
)
from mirascope.tools import DuckDuckGoSearch, ParseURLContent

class WebAssistant(BaseModel):
    messages: list[BaseMessageParam] = []
    search_history: list[str] = []
    max_results_per_query: int = 2

    @openai.call(model="gpt-4o-mini", stream=True)
    @prompt_template(
        """
        SYSTEM:
        You are an expert web searcher. Your task is to answer the user's question using the provided tools.
        The current date is {current_date}.

        You have access to the following tools:
        - `_web_search`: Search the web when the user asks a question. Follow these steps for EVERY web search query:
            1. There is a previous search context: {self.search_history}
            2. There is the current user query: {question}
            3. Given the previous search context, generate multiple search queries that explores whether the new query might be related to or connected with the context of the current user query. 
                Even if the connection isn't immediately clear, consider how they might be related.
        - `extract_content`: Parse the content of a webpage.

        When calling the `_web_search` tool, the `body` is simply the body of the search
        result. You MUST then call the `extract_content` tool to get the actual content
        of the webpage. It is up to you to determine which search results to parse.

        Once you have gathered all of the information you need, generate a writeup that
        strikes the right balance between brevity and completeness based on the context of the user's query.

        MESSAGES: {self.messages}
        USER: {question}
        """
    )
    def _stream(self, question: str) -> openai.OpenAIDynamicConfig:
        return {
            "tools": [DuckDuckGoSearch, ParseURLContent],
            "computed_fields": {
                "current_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            },
        }

    def _step(self, question: str):
        response = self._stream(question)
        tools_and_outputs = []
        for chunk, tool in response:
            if tool:
                print(f"\nUsing {tool._name()} tool with args: {tool.args}")
                tools_and_outputs.append((tool, tool.call()))
            else:
                print(chunk.content, end="", flush=True)
        
        if response.user_message_param:
            self.messages.append(response.user_message_param)
        self.messages.append(response.message_param)
        
        if tools_and_outputs:
            self.messages += response.tool_message_params(tools_and_outputs)
            self._step("")

    def run(self):
        while True:
            question = input("\n(User): ")
            if question == "exit":
                break
            print("(Assistant): ", end="", flush=True)
            self._step(question)

# Initialize and run
if __name__ == "__main__":
    WebAssistant().run() 