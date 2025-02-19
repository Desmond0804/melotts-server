import os
import tempfile
import uvicorn
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from dotenv import load_dotenv
from contextlib import asynccontextmanager
from melo.api import TTS

load_dotenv()
TTS_LANGUAGE = os.getenv("TTS_LANGUAGE", "EN") # EN -> English, ES -> Spanish, FR -> French, ZH -> Chinese, JP -> Japanese, KR -> Korean
TTS_RESPONSE_FORMAT = os.getenv("TTS_RESPONSE_FORMAT", "mp3")
TTS_SPEED = float(os.getenv("TTS_SPEED", 1.0))

device = "auto"
model = None

@asynccontextmanager
async def lifespan():
    global model
    global speaker_ids
    # load TTS model
    model = TTS(language=TTS_LANGUAGE, device=device)
    speaker_ids = model.hps.data.spk2id
    yield
    # clean up TTS model & release resources
    model.close()

class TTSRequest(BaseModel):
    model: str = "tts-1" # or "tts-1-hd"
    input: str
    voice: str = "EN-Default"  # EN-US, EN-BR, EN-INDIA, EN-AU, EN-Default, ES, FR, ZH, JP, KR
    response_format: str = TTS_RESPONSE_FORMAT # mp3, opus, aac, flac, wav
    speed: float = TTS_SPEED # 0.25 - 4.0

app = FastAPI(lifespan=lifespan)

@app.post("/v1/audio/speech", response_class=StreamingResponse)
async def generate_speech(request: TTSRequest):
    response_format = request.response_format
    # set the Content-Type header based on the requested format
    if response_format == "mp3":
        media_type = "audio/mpeg"
    elif response_format == "opus":
        media_type = "audio/ogg;codec=opus"
    elif response_format == "aac":
        media_type = "audio/aac"
    elif response_format == "flac":
        media_type = "audio/x-flac"
    elif response_format == "wav":
        media_type = "audio/wav"

    # generate speech & save to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{request.response_format}") as tmp:
        output_path = tmp.name
        model.tts_to_file(request.input, speaker_id=speaker_ids[request.voice], output_path=output_path, speed=request.speed)
    
    def generate():
        with open(output_path, mode="rb") as audio_file:
            yield from audio_file

    return StreamingResponse(content=generate(), media_type=media_type)


if __name__ == "__main__":
    uvicorn.run(app)