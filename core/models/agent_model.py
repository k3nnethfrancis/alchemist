"""
agent_model.py

This module defines the AgentModel class, which interacts with a language model
to generate responses.
"""

from typing import Optional

import openai


class AgentModel:
    """
    Represents the language model used by the agent to generate responses.
    """

    def __init__(self, model_name: str = "gpt-3.5-turbo"):
        """
        Initializes the AgentModel.

        Args:
            model_name (str): The name of the model to use.
        """
        self.model_name = model_name
        # Set your OpenAI API key
        openai.api_key = "YOUR_OPENAI_API_KEY"

    async def generate_response(self, prompt: str) -> Optional[str]:
        """
        Generates a response from the model based on the prompt.

        Args:
            prompt (str): The input prompt for the language model.

        Returns:
            Optional[str]: The generated response, or None if unsuccessful.
        """
        try:
            response = openai.Completion.create(
                engine=self.model_name,
                prompt=prompt,
                max_tokens=150,
                n=1,
                stop=None,
                temperature=0.7,
            )
            return response.choices[0].text.strip()
        except Exception as e:
            # Handle exceptions (e.g., API errors)
            print(f"Error generating response: {e}")
            return None