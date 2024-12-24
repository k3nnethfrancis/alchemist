from setuptools import setup, find_packages

setup(
    name="ai",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "discord.py",
        "python-dotenv",
        "pydantic",
        "openai",
        "anthropic",
        "pytest",
        "pytest-asyncio",
    ],
    python_requires=">=3.8",
) 