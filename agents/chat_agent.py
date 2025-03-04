### A PROVIDER AGNOSTIC CHATBOT ###

import json
from pydantic import BaseModel
from mirascope.core import BaseDynamicConfig, Messages, openai

class Book(BaseModel):
    title: str
    author: str

class Librarian(BaseModel):
    history: list[openai.OpenAIMessageParam] = []
    library: list[Book] = [
        Book(title="The Name of the Wind", author="Patrick Rothfuss"),
        Book(title="Mistborn: The Final Empire", author="Brandon Sanderson"),
    ]

    def _available_books(self) -> str:
        """Returns the list of books available in the library."""
        return json.dumps([book.model_dump() for book in self.library])

    @openai.call("gpt-4o-mini", stream=True)
    def _stream(self, query: str) -> BaseDynamicConfig:
        messages = [
            Messages.System("You are a librarian"),
            *self.history,
            Messages.User(query),
        ]
        return {"messages": messages, "tools": [self._available_books]}

    def _step(self, query: str) -> None:
        if query:
            self.history.append(Messages.User(query))
        stream = self._stream(query)
        tools_and_outputs = []
        for chunk, tool in stream:
            if tool:
                print(f"\nUsing {tool._name()} tool with args: {tool.args}")
                tools_and_outputs.append((tool, tool.call()))
            else:
                print(chunk.content, end="", flush=True)
        
        self.history.append(stream.message_param)
        if tools_and_outputs:
            self.history += stream.tool_message_params(tools_and_outputs)
            self._step("")

    def run(self):
        while True:
            query = input("\n(User): ")
            if query == "exit":
                break
            print("(Assistant): ", end="", flush=True)
            self._step(query)

if __name__ == "__main__":
    Librarian().run()
