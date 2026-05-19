from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pathlib import Path

from app.routes import predict, indicators, train

app = FastAPI(title="Stock AI V2", version="2.0.0")

app.include_router(predict.router,    prefix="/api")
app.include_router(indicators.router, prefix="/api")
app.include_router(train.router,      prefix="/api")

FRONTEND = Path(__file__).parent.parent / "frontend"
app.mount("/static", StaticFiles(directory=FRONTEND), name="static")

@app.get("/", include_in_schema=False)
def root():
    return FileResponse(FRONTEND / "index.html")

@app.get("/health")
def health():
    return {"status": "ok", "version": "2.0.0"}
