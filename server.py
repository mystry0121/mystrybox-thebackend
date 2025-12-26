import os, uuid
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import replicate

app = FastAPI()
JOBS = {}

class GenerateRequest(BaseModel):
    prompt: str
    duration_sec: int = 8

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/v1/generate")
def generate(req: GenerateRequest):
    if not os.getenv("REPLICATE_API_TOKEN"):
        raise HTTPException(500, "Missing REPLICATE_API_TOKEN")

    job_id = f"job_{uuid.uuid4().hex[:10]}"
    JOBS[job_id] = {"status": "running"}

    try:
        output = replicate.run(
            "meta/musicgen:671ac645ce5e552cc63a54a2bbff63fcf798043055d2dac5fc9e36a837eedcfb",
            input={
                "prompt": req.prompt,
                "duration": req.duration_sec
            }
        )

        # Force output into a plain string for JSON encoding
        if isinstance(output, list) and len(output) > 0:
            audio_url = str(output[0])
        else:
            audio_url = str(output)

        JOBS[job_id]["status"] = "succeeded"
        JOBS[job_id]["audio_url"] = audio_url

        return {
            "job_id": job_id,
            "status": "succeeded",
            "audio_url": audio_url
        }

    except Exception as e:
        JOBS[job_id]["status"] = "failed"
        JOBS[job_id]["error"] = str(e)
        return {
            "job_id": job_id,
            "status": "failed",
            "error": str(e)
        }
