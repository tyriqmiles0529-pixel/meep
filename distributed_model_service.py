import fastapi
import uvicorn
from pydantic import BaseModel
from typing import List, Optional
from meep_terminal.core.engine import TerminalEngine
import threading

app = fastapi.FastAPI(title="MEEP Inference Service")
engine = TerminalEngine()

class InferenceRequest(BaseModel):
    date_str: str
    explain: bool = True

class TaskStatus(BaseModel):
    task_id: int
    status: str
    progress: float
    message: str

@app.post("/inference/run")
async def run_inference(req: InferenceRequest):
    """
    Decoupled Inference Endpoint.
    Launches the neural core in a background thread to prevent API blocking.
    """
    task_id = engine.start_background_inference(req.date_str)
    return {"task_id": task_id, "status": "queued"}

@app.get("/tasks/{task_id}")
async def get_task(task_id: int):
    status = engine.get_task_status(task_id)
    return status

if __name__ == "__main__":
    print("🚀 MEEP Distributed Inference Service Starting...")
    uvicorn.run(app, host="0.0.0.0", port=8080)
