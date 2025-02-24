from fastapi import FastAPI
from agent import process_query
from schemas import Query

app = FastAPI()

@app.post("/ask")
def ask_question(query: Query):
    response, chart = process_query(query.question)
    return {"answer": response, 'chart': chart}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)