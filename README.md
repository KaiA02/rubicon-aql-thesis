# Rubicon: AQL-Based Multi-Source Retrieval System

This repository contains the implementation and evaluation artifacts for the bachelor's thesis:

**"Building and Evaluating a Multi-Source Query Interface with Unified Operators"**

## Overview

Rubicon introduces an operator-based query language (AQL) for structured and transparent retrieval across heterogeneous data sources such as databases, email systems, and web resources.

This repository provides:
- the system implementation,
- evaluation tasks and gold-standard definitions,
- anonymized user study results
- langchain results

## Tech Stack

- **Backend:** Python, Flask, Flask-Login, Flask-Bcrypt
- **LLM & Retrieval:** OpenAI API, RAG with Sentence Transformers
- **Data sources:** SQLite (university data warehouse), Wikipedia API, Gmail API (OAuth 2.0), web scraping (BeautifulSoup)
- **Core components:** custom AQL parser, source-specific wrappers
- **Evaluation:** user study (pre-, main- and post-study), comparison with LangChain ReAct agents

## Related work

Parts of this implementation and evaluation were used in *An Alternate Agentic AI Architecture (It's About the Data)*, [arXiv:2604.21413v1](https://arxiv.org/abs/2604.21413v1).

## Repository Structure
code/ Implementation of Rubicon and the AQL interface
study/ Pre-study, main-study and post-study materials
tasks/ Task descriptions and gold answers
langchain.pdf
README.md

## Reproducibility

All evaluation results reported in the thesis are based on the data provided in this repository. Participant data has been anonymized.

## Author

Kai Arenja
