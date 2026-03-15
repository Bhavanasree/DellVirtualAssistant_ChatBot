# Dell RAG Assistant

A Retrieval-Augmented Generation (RAG) chatbot for Dell laptop recommendations and specifications, built as a capstone project.

## Overview

This project implements a RAG system that allows users to query Dell laptop information extracted from PDF documents. The system uses AWS Bedrock for language models, ChromaDB for vector storage, and provides both a backend API and a Streamlit web interface.

## Features

- **PDF Processing**: Extracts text and tables from Dell laptop specification PDFs
- **Vector Search**: Uses embeddings to find relevant information
- **LLM Integration**: Powered by Anthropic Claude via AWS Bedrock
- **Reranking**: Implements Cohere reranking for improved retrieval
- **Web Interface**: Streamlit-based frontend for easy interaction
- **REST API**: FastAPI backend for programmatic access
- **Evaluation**: RAGAS-based evaluation metrics for system performance

## Tech Stack

- **Backend**: Python, FastAPI, LangChain
- **Frontend**: Streamlit
- **LLM**: Anthropic Claude 3 Haiku (AWS Bedrock)
- **Embeddings**: Amazon Titan (AWS Bedrock)
- **Vector Database**: ChromaDB
- **Reranking**: Cohere Rerank
- **Evaluation**: RAGAS
- **PDF Processing**: pdfplumber

## Prerequisites

- Python 3.12+
- AWS Account with Bedrock access
- AWS CLI configured with appropriate credentials
- Virtual environment (recommended)

## Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd dell-rag-assistant
   ```

2. **Create and activate virtual environment**:
   ```bash
   python -m venv ragchatbot
   ragchatbot\Scripts\activate  # Windows
   # or
   source ragchatbot/bin/activate  # Linux/Mac
   ```

3. **Install dependencies**:
   ```bash
   cd backend
   pip install -r requirements.txt
   ```

4. **Set up environment variables**:
   Create a `.env` file in the backend directory:
   ```
   PDF_FOLDER=pdfs
   # AWS credentials should be configured via AWS CLI or environment variables
   ```

5. **Place PDF files**:
   Add Dell laptop specification PDFs to the `pdfs/` directory.

## Usage

### Backend API

1. **Start the FastAPI server**:
   ```bash
   cd backend
   uvicorn main:app --port 8000
   ```

2. **API Endpoints**:
   - `GET /`: Health check
   - `POST /chat`: Send chat messages

### Frontend Interface

1. **Start the Streamlit app**:
   ```bash
   cd frontend
   streamlit run streamlit_app.py
   ```

2. **Access the interface**:
   Open http://localhost:8501 in your browser

### Evaluation

1. **Run evaluation**:
   ```bash
   cd backend
   python evaluation/evaluation_runner.py
   ```

## Project Structure

```
dell-rag-assistant/
├── backend/
│   ├── main.py                 # FastAPI application
│   ├── rag_pipeline.py         # RAG pipeline implementation
│   ├── requirements.txt        # Python dependencies
│   ├── __init__.py
│   └── evaluation/
│       ├── evaluation_runner.py    # Evaluation script
│       ├── ragas_evaluator.py      # RAGAS evaluation
│       ├── dataset_builder.py      # Dataset creation
│       └── __init__.py
├── frontend/
│   └── streamlit_app.py        # Streamlit interface
├── pdfs/                       # PDF documents directory
├── chroma_db/                  # Vector database storage
├── ragchatbot/                 # Virtual environment
└── README.md                   # This file
```

## Configuration

### AWS Bedrock Setup

1. Ensure your AWS account has access to Bedrock
2. Enable the following models:
   - `anthropic.claude-3-haiku-20240307-v1:0`
   - `amazon.titan-embed-text-v2:0`
3. Configure AWS credentials:
   ```bash
   aws configure
   ```

### Environment Variables

- `PDF_FOLDER`: Path to PDF files (default: `pdfs`)

## Evaluation Metrics

The system uses RAGAS to evaluate:
- **Faithfulness**: How well the answer matches the retrieved context
- **Answer Relevancy**: How relevant the answer is to the question
- **Context Precision**: Precision of retrieved contexts
- **Context Recall**: Recall of retrieved contexts

## Development

### Building the Vector Database

The system automatically builds the vector database on first run, but you can rebuild it by:

1. Deleting the `chroma_db/` directory
2. Running the evaluation or API (it will rebuild automatically)

### Adding New PDFs

1. Place new PDF files in the `pdfs/` directory
2. Rebuild the vector database as described above

## Troubleshooting

### Common Issues

1. **Model Timeout**: Increase timeout in `rag_pipeline.py` if requests are timing out
2. **Memory Issues**: Reduce chunk size in text splitting if encountering memory errors
3. **PDF Processing**: Ensure PDFs are text-based, not image-based

### Logs

Check terminal output for detailed error messages and processing status.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Built as a capstone project for demonstrating RAG capabilities
- Uses open-source libraries and cloud services
- Inspired by modern AI assistant architectures