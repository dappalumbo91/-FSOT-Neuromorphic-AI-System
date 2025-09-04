#!/usr/bin/env python3

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="fsot-neuromorphic-ai",
    version="1.0.0",
    author="FSOT AI Development Team",
    author_email="contact@fsot-ai.com",
    description="Complete Human Brain-Inspired AI System with Neuromorphic Architecture",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/dappalumbo91/FSOT-Neuromorphic-AI-System",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-asyncio>=0.19.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "mypy>=0.961",
        ],
        "gpu": [
            "torch>=1.11.0+cu117",
            "torchvision>=0.12.0+cu117",
        ],
        "all": [
            "pytest>=7.0.0",
            "pytest-asyncio>=0.19.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "mypy>=0.961",
            "torch>=1.11.0+cu117",
            "torchvision>=0.12.0+cu117",
        ],
    },
    entry_points={
        "console_scripts": [
            "fsot-brain=src.neuromorphic_brain.main:main",
            "fsot-demo=examples.basic_usage:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
