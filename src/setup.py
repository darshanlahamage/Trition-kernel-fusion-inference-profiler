# setup.py
from setuptools import setup, find_packages

setup(
    name="triton_llm_engine",
    version="0.1.0",
    description="Custom high-performance LLM primitives using OpenAI Triton.",
    packages=find_packages(),
    install_requires=[
        "torch",
        "triton",
        "pandas",
        "matplotlib"
    ],
    python_requires=">=3.10",
)