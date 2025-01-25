from setuptools import setup, find_packages

setup(
    name="alchemist",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "pydantic",
        "pytest",
        "pytest-asyncio",
        "mirascope"
    ],
    python_requires=">=3.8",
    # Add metadata for PyPI
    author="kenneth cavanagh",
    author_email="ken@agency42.com",
    description="A python library for building AI social agents",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/k3nnethfrancis/alchemist",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
    ],
)