from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import uvicorn
from connect_memory_with_llm_lighter_cos import (
    get_vectorstore,
    get_embeddings,
    retrieve_relevant_docs,
    OpenRouterLLM,
    CustomRetriever,
    RetrievalQA,
    french_course_prompt
)
import os

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize the vectorstore and embeddings
db = get_vectorstore()
embeddings = get_embeddings()

class ChatRequest(BaseModel):
    message: str
    similarity_threshold: float = 0.35
    max_docs: int = 5
    filters: Optional[dict] = None

class ChatResponse(BaseModel):
    response: str

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        # Initialize LLM
        llm = OpenRouterLLM(
            api_key= "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",
            model="meta-llama/llama-4-scout:free",
        )

        # Get relevant documents
        search_db = db
        if request.filters:
            # Apply filters if provided
            from connect_memory_with_llm_lighter_cos import filter_docs
            search_db = filter_docs(db, request.filters)

        relevant_docs = retrieve_relevant_docs(
            search_db,
            request.message,
            embeddings,
            threshold=request.similarity_threshold,
            top_k=request.max_docs
        )

        # Create retriever and QA chain
        retriever = CustomRetriever(documents=relevant_docs)
        qa = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs={
                "prompt": french_course_prompt(),
            },
        )

        # Get response
        result = qa({"query": request.message})
        answer = result["result"]

        return ChatResponse(
            response=answer
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/accueil")
async def accueil():
    return {"message": "Bienvenue sur la page d'accueil!"}

if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)