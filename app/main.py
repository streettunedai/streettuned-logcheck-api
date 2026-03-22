from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def root():
    return {"message": "StreetTunedAI LogCheck API is running"}

@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/validate")
def validate():
    return {"status": "ready", "platform": "ls_gas"}