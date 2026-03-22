from fastapi import FastAPI, UploadFile, File

app = FastAPI()

@app.get("/")
def root():
    return {"message": "StreetTunedAI LogCheck API is running"}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/validate")
async def validate(file: UploadFile = File(...)):
    contents = await file.read()
    size_bytes = len(contents)

    return {
        "status": "ready",
        "platform": "ls_gas",
        "filename": file.filename,
        "content_type": file.content_type,
        "size_bytes": size_bytes
    }