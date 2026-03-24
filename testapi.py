from fastapi import FastAPI, UploadFile, File, HTTPException
import whisper
import shutil
import os
import uvicorn
import uuid
import httpx
import tempfile

app = FastAPI()

# Load model once
model = whisper.load_model("small")


@app.get("/")
def root():
    return {"message": "Whisper API is running 🚀"}


# -------------------------------
# 🔹 Upload File API
# -------------------------------
@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):
    try:
        # Create unique temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp:
            temp_file = temp.name

            # Save uploaded file
            shutil.copyfileobj(file.file, temp)

        # Transcribe
        result = model.transcribe(temp_file, fp16=False)

        return {
            "filename": file.filename,
            "text": result["text"]
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        # Cleanup
        if 'temp_file' in locals() and os.path.exists(temp_file):
            os.remove(temp_file)


# -------------------------------
# 🔹 S3 URL API (Async + Streaming)
# -------------------------------
@app.post("/transcribe-from-url")
async def transcribe_from_url(s3_url: str):
    # Basic validation (very important)
    if not s3_url.startswith("http"):
        raise HTTPException(status_code=400, detail="Invalid URL")

    try:
        # Create temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp:
            temp_file = temp.name

        # Async download with streaming
        async with httpx.AsyncClient(timeout=60.0) as client:
            async with client.stream("GET", s3_url) as response:
                if response.status_code != 200:
                    raise HTTPException(status_code=400, detail="Failed to download file")

                with open(temp_file, "wb") as f:
                    async for chunk in response.aiter_bytes():
                        f.write(chunk)

        # Transcribe
        result = model.transcribe(temp_file, fp16=False)

        return {
            "text": result["text"]
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        # Cleanup
        if 'temp_file' in locals() and os.path.exists(temp_file):
            os.remove(temp_file)


# -------------------------------
# 🔹 Run Server
# -------------------------------
if __name__ == "__main__":
    uvicorn.run("testapi:app", host="127.0.0.1", port=8000, reload=True)