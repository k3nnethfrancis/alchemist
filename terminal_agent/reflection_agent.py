from typing import Literal
from datetime import datetime
from pydantic import BaseModel, Field
from mirascope.core import (
    BaseMessageParam,
    Messages,
    openai,
)
import time

class ReflectionAgent(BaseModel):
    history: list[dict] = []

    @openai.call("gpt-4o-mini")
    def _step_response(self, prompt: str, step_number: int, previous_steps: str) -> str:
        messages = [
            Messages.System("""You are an expert AI assistant that explains your reasoning step by step.
                For this step, provide your response in this exact format:
                TITLE: [title of the step]
                CONTENT: [your detailed reasoning]
                NEXT: [either "continue" or "final_answer"]

                Guidelines:
                - Use AT MOST 5 steps to derive the answer.
                - Be aware of your limitations as an LLM and what you can and cannot do.
                - In your reasoning, include exploration of alternative answers.
                - Consider you may be wrong, and if you are wrong in your reasoning, where it would be.
                - Fully test all other possibilities.
                - YOU ARE ALLOWED TO BE WRONG. When you say you are re-examining
                    - Actually re-examine, and use another approach to do so.
                    - Do not just say you are re-examining.

                This is step number {step_number}.
                """),
            Messages.User(f"""Question: {prompt}

                Previous steps:
                {previous_steps}""")
        ]
        return {"messages": messages}

    @openai.call("gpt-4o-mini")
    def _final_answer(self, prompt: str, reasoning: str) -> str:
        messages = [
            Messages.System("""Based on the following chain of reasoning, provide a final answer to the question.
                Only provide the text response without any titles or preambles."""),
            Messages.User(f"""Question: {prompt}

                Reasoning:
                {reasoning}

                Final Answer:""")
        ]
        return {"messages": messages}

    def _parse_step_response(self, response: str) -> tuple[str, str, str]:
        """Parse the step response into title, content, and next action."""
        parts = response.split('\n')
        title = ""
        content = ""
        next_action = "continue"
        
        for part in parts:
            part = part.strip()
            if part.startswith("TITLE:"):
                title = part[6:].strip()
            elif part.startswith("CONTENT:"):
                content = part[8:].strip()
            elif part.startswith("NEXT:"):
                next_action = part[5:].strip()
        
        return title, content, next_action

    def _generate_response(self, query: str) -> tuple[list[tuple[str, str, float]], float]:
        steps: list[tuple[str, str, float]] = []
        total_thinking_time: float = 0.0
        step_count: int = 1
        reasoning: str = ""
        previous_steps: str = ""

        print("\nThinking...\n", flush=True)
        
        while True:
            # Start timing
            start_time = datetime.now()
            
            # Make the API call and get response
            response = self._step_response(query, step_count, previous_steps)
            
            # End timing
            end_time = datetime.now()
            thinking_time = (end_time - start_time).total_seconds()

            # Parse the response
            title, content, next_action = self._parse_step_response(response.content)
            
            # Print results with forced flushing
            print(f"{title}", flush=True)
            print(f"{content}", flush=True)
            print(f"[Thinking time: {thinking_time:.2f} seconds]\n", flush=True)
            
            # Store results
            steps.append(
                (title, content, thinking_time)
            )
            total_thinking_time += thinking_time

            reasoning += f"\n{content}\n"
            previous_steps += f"\n{content}\n"

            if next_action == "final_answer" or step_count >= 5:
                break

            step_count += 1

            # Add a small delay to ensure output is visually distinct
            time.sleep(0.1)

        print("Generating final answer...", end="", flush=True)
        start_time = datetime.now()
        final_result = self._final_answer(query, reasoning)
        end_time = datetime.now()
        thinking_time = (end_time - start_time).total_seconds()
        total_thinking_time += thinking_time

        print(f"\n{final_result.content}")
        print(f"[Thinking time: {thinking_time:.2f} seconds]\n")
        
        steps.append(("Final Answer", final_result.content, thinking_time))
        return steps, total_thinking_time

    def run(self) -> None:
        while True:
            query = input("\n(User): ")
            if query.lower() in ["exit", "quit"]:
                break

            print("(Assistant): ", end="", flush=True)
            steps, total_time = self._generate_response(query)
            print(f"[Total thinking time: {total_time:.2f} seconds]")

            self.history.append({"role": "user", "content": query})
            self.history.append({"role": "assistant", "content": steps[-1][1]})

if __name__ == "__main__":
    ReflectionAgent().run()