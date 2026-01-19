"""
Setup configuration for BondTrader package
"""

from setuptools import find_packages, setup

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="bondtrader",
    version="1.0.0",
    author="Sage Hart",
    description="Comprehensive Bond Trading & Arbitrage Detection System",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/bondtrader",
    packages=find_packages(exclude=["tests", "docs", "scripts"]),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Financial and Insurance Industry",
        "Topic :: Office/Business :: Financial :: Investment",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.9",  # Python 3.8 EOL - many dependencies require 3.9+
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black",
            "flake8",
            "mypy",
        ],
    },
    # Note: Scripts are in scripts/ directory, not part of package
    # Run with: streamlit run scripts/dashboard.py
)
