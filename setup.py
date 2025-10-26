"""
Setup script for SOREModel
Instalação e configuração do projeto
"""

from setuptools import setup, find_packages
import os

# Ler README
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Ler requirements
with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="soremodel",
    version="2.0.0",
    author="SOREModel Team",
    author_email="soremodel@gmail.com",
    description="Simple Open-Source Recurrent/Transformer Model for text generation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Elitinho123456/SOREModel",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Text Processing :: Linguistic",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.2.4",
            "pytest-cov>=2.12.1",
            "mypy>=0.910",
            "black>=21.0.0",
            "flake8>=3.9.0",
        ],
        "docs": [
            "sphinx>=4.0.0",
            "sphinx-rtd-theme>=0.5.2",
        ],
    },
    entry_points={
        "console_scripts": [
            "soremodel-train=scripts.train:main",
            "soremodel-generate=scripts.generate:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
