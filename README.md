# Backstory Consistency NLP (KDSH)

This project is a Retrieval Augmented Generation (RAG) system designed to check the consistency of character backstories against a corpus of books.

## Project Structure

- `Code.ipynb`: The original Jupyter Notebook for development and experimentation.
- `Dataset/`: Contains the training data (`train.csv`), test data (`test.csv`), and book texts in the `Books/` directory.
- `data_ingestion.py`: Handles loading and embedding of book texts using Pathway and SentenceTransformers.
- `query_generator.py`: Uses an LLM to decompose backstories into verifiable claims and generate search queries.
- `evidence_retrieval.py`: Retrieves relevant evidence chunks from the embedded books using a cross-encoder reranker.
- `verification.py`: Uses an LLM to verify claims against the retrieved evidence.
- `validator.py`: Orchestrates the entire validation pipeline.
- `results.csv`: Output of the validation process.

## Prerequisites

- Python 3.10+
- Dependencies listed in `requirements.txt`

## Setup

1.  Clone the repository:
    ```bash
    git clone https://github.com/sahilgupta630/KDSH.git
    cd KDSH
    ```

2.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3.  Set up Environment Variables:
    Create a `.env` file or set the following environment variables:
    - `GROQ_API_KEY`: Your Groq API key for the LLM.

## Usage

You can run the pipeline by executing the `Code.ipynb` notebook or by importing the modules in your own script.

The `validator.py` script contains the main `run_validation` function which takes the LLM client, model name, and the indexed data frame as input.

```python
import os
from groq import Groq
from data_ingestion import NovelIngestionPipeline
from validator import run_validation

# Setup
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
model_name = "llama3-70b-8192" # or similar

# Ingest Data
pipeline = NovelIngestionPipeline("Dataset/Books")
index_df = pipeline.run_indexing()

# Run Validation
results = run_validation(client, model_name, index_df, limit=10)
print(results)
```
