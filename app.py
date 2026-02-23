from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Medical Image Diagnosis API (FastAPI) is up and running!"}