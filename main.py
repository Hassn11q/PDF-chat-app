import os
from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from dotenv import load_dotenv
import logging
from pathlib import Path
from langchain_cohere import ChatCohere, CohereEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.document_loaders import PyPDFium2Loader
from langchain_chroma import Chroma
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import ChatPromptTemplate
from langchain.schema import Document

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="AI PDF Chat Assistant", version="2.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Pydantic models
class ChatRequest(BaseModel):
    query: str

class UploadResponse(BaseModel):
    status: str
    filename: str
    pages: int
    chunks: int

class ChatResponse(BaseModel):
    answer: str

# Global variables
COHERE_API_KEY = os.getenv("COHERE_API_KEY")
if not COHERE_API_KEY:
    raise ValueError("Cohere API key is missing. Please set it in the .env file.")

# Initialize LLM and Embeddings
llm = ChatCohere(model="command-r-plus", cohere_api_key=COHERE_API_KEY)
embeddings = CohereEmbeddings(model="embed-multilingual-light-v3.0", cohere_api_key=COHERE_API_KEY)

# Text splitters
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=80)
semantic_splitter = SemanticChunker(embeddings=embeddings, breakpoint_threshold_type="interquartile")

# Prompt template
PROMPT_TEMPLATE = """
<|system|>
You are a research assistant. Use the following context to answer the user's query in a structured format with detailed citations. Your response should be in Arabic if the query is in Arabic, otherwise in English.

IMPORTANT RULES:
1. Start your response with "Query Result:" followed by a concise summary answering the query, using 2-3 sentences maximum.
2. After the summary, list exactly 2 citations under "Citations:".
3. Each citation should be numbered and in the format: [number]Document Title(Page: X): Exact quote from the document
4. Citations MUST be VERBATIM and EXACT quotes from the provided context.
5. Include page numbers for each citation if available.
6. DO NOT use ellipses (...) or any other shortening techniques in citations.
7. DO NOT paraphrase or modify the original text in any way for citations.
8. Ensure citations are substantial and provide meaningful context.
9. The first citation should always be the title of the document from the first page.
10. The second citation should be a relevant quote from the document, preferably from a later page.

CONTEXT: {context}
</s> 
<|user|>
{query}
</|assistant|>
Query Result:
[Provide a concise summary answering the query in 2-3 sentences, referencing the citations.]

Citations:
[1]Document Title(Page: 1): Exact title of the document
[2]Document Title(Page: X): Relevant quote from the document
"""

prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)

# Dependency for vector store
def get_vector_store():
    return Chroma(persist_directory="./db", embedding_function=embeddings)

# Routes
@app.get("/", response_class=HTMLResponse)
async def read_index(request: Request):
    with open("static/index.html", "r") as f:
        content = f.read()
    return HTMLResponse(content=content)

@app.get("/health")
async def health_check():
    return {"status": "ok"}

@app.post("/upload_pdf", response_model=UploadResponse)
async def upload_file(file: UploadFile = File(...)):
    try:
        logger.info(f"Received file: {file.filename}")
        
        file_name = Path(file.filename).name
        save_path = Path("data") / file_name
        save_path.parent.mkdir(exist_ok=True)
        
        content = await file.read()
        logger.info(f"File size: {len(content)} bytes")
        save_path.write_bytes(content)
        
        loaded_documents = PyPDFium2Loader(str(save_path)).load()
        document_text = "\n".join(doc.page_content for doc in loaded_documents)
        documents = text_splitter.split_documents(semantic_splitter.create_documents([document_text]))
        
        vectorstore = get_vector_store()
        vectorstore.add_documents(documents)
        
        return UploadResponse(
            status="Successfully Uploaded and Processed",
            filename=file_name,
            pages=len(loaded_documents),
            chunks=len(documents)
        )
    except Exception as e:
        logger.error(f"Error in upload_file: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat_pdf", response_model=ChatResponse)
async def chat_pdf(request: ChatRequest, vectorstore: Chroma = Depends(get_vector_store)):
    try:
        formatted_chunks = [
            chunk.page_content if isinstance(chunk, Document) else str(chunk)
            for chunk in vectorstore.get()["documents"]
        ]
        keyword_retriever = BM25Retriever.from_texts(formatted_chunks)
        keyword_retriever.k = 3
        vectorstore_retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
        ensemble_retriever = EnsembleRetriever(
            retrievers=[vectorstore_retriever, keyword_retriever], weights=[0.5, 0.5])
        
        chain = (
            {"context": ensemble_retriever, "query": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )
        response = chain.invoke(request.query)
        
        logger.info(f"Raw response from LLM: {response}")
        
        parts = response.split("Citations:", 1)
        query_result = parts[0].replace("Query Result:", "").strip()
        
        if len(parts) > 1:
            citations_text = parts[1].strip()
            citations = [line.strip() for line in citations_text.split('\n') if line.strip()]
        else:
            citations = []

        while len(citations) < 2:
            citations.append("[X]Document Title(Page: X): No additional citation available")

        formatted_response = f"Query Result:\n\n{query_result}\n\nCitations:\n\n"
        for i, citation in enumerate(citations[:2], 1):
            formatted_response += f"[{i}] {citation}\n\n"
        
        return ChatResponse(answer=formatted_response)
    except Exception as e:
        logger.error(f"Error in chat_pdf: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))