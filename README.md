# AI PDF Chat Assistant

## Overview

The **AI PDF Chat Assistant** is a FastAPI application that allows users to upload PDF documents and interact with them through a chat interface. It uses advanced language models and embeddings to process and retrieve information from the uploaded PDFs, providing structured responses with citations.

## Features

- Upload PDF files for processing.
- Chat with the uploaded PDF to get answers to queries.
- Supports multiple languages (Arabic and English).
- Provides citations from the PDF documents in responses.
- Built with FastAPI and Langchain.

## Getting Started

### Prerequisites

- Python 3.10 or higher
- An active [Cohere API](https://cohere.ai/) account (to obtain your API key)

### Installation

1. Clone the repository:

   ```bash
   git clone <repository-url>
   cd ai-pdf-chat-assistant
   ```

2. Create a virtual environment:

```bash
 python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```
3. Install the required packages:

```bash
pip install -r requirements.txt
```
4. Create a .env file in the project root directory and add your Cohere API key:

```bash
COHERE_API_KEY=your_cohere_api_key
PORT=8000  # Optional: set the port number
```

## Running the Application
To start the FastAPI application, run the following command:
```bash
uvicorn main:app --host 0.0.0.0 --port $PORT
```
## Accessing the Application
Once the server is running, you can access the application at http://localhost:8000. You can also access the interactive API documentation at http://localhost:8000/docs.




# PDF-chat-app
