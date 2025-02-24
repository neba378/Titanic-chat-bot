from fastapi import FastAPI # type: ignore
from agent import process_query
from schemas import Query
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def ask_question():
    return {"message": "Welcome to the Titanic Chatbot API! Please make a POST request to the /ask endpoint with a JSON body containing your question."}

@app.post("/ask")
def ask_question(query: Query):
    response, chart = process_query(query.question)
    return {"answer": response, 'chart': chart}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
