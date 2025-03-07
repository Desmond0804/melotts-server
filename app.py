import os
import tempfile
import uvicorn
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from dotenv import load_dotenv
from contextlib import asynccontextmanager
from melo.api import TTS
from langdetect import detect

load_dotenv()
TTS_LANGUAGE = os.getenv("TTS_LANGUAGE", "EN") # EN -> English, ES -> Spanish, FR -> French, ZH_MIX_EN -> Chinese, JP -> Japanese, KR -> Korean
TTS_VOICE = os.getenv("TTS_VOICE", "EN-Default") # EN-US, EN-BR, EN-INDIA, EN-AU, EN-Default, ES, FR, ZH, JP, KR
TTS_RESPONSE_FORMAT = os.getenv("TTS_RESPONSE_FORMAT", "mp3") # mp3, opus, aac, flac, wav
TTS_SPEED = float(os.getenv("TTS_SPEED", 1.0)) # 0.25 - 4.0

device = "auto"
model = None
supported_languages = ["en", "es", "fr", "ja", "ko", "zh-cn", "zh-tw", "ms", "id"]

# get TTS model for MS language
from huggingface_hub import hf_hub_download
ckpt_path = hf_hub_download(repo_id='mesolitica/MeloTTS-MS', filename='model.pth')
config_path = hf_hub_download(repo_id='mesolitica/MeloTTS-MS', filename='config.json')

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    global speaker_ids
    # load TTS model
    if TTS_LANGUAGE == "MS":
        model = TTS(language=TTS_LANGUAGE, device=device, config_path=config_path, ckpt_path=ckpt_path)
    else:
        model = TTS(language=TTS_LANGUAGE, device=device)
    speaker_ids = model.hps.data.spk2id
    yield
    # clean up TTS model & release resources
    del model

class TTSRequest(BaseModel):
    model: str = "tts-1" # or "tts-1-hd"
    input: str
    voice: str = TTS_VOICE
    response_format: str = TTS_RESPONSE_FORMAT
    speed: float = TTS_SPEED

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

    language = detect(request.input)
    print(f"*****Language Detected: {language}*****")
    if language not in supported_languages:
        language = "EN"
    else: # match the langauge codes of langdetect to MeloTTS
        if language == "zh-cn" or language == "zh-tw":
            language = "ZH"
        elif language == "ja":
            language = "JP"
        elif language == "ko":
            language = "KR"
        elif language == "id":
            language = "MS"
        else:
            language = language.upper()
    
    # set the voice based on the langauge
    voice = language
    if language == "EN":
        voice = "EN-Default"
    if language == "MS":
        voice = "husein-chatbot"

    global model
    global speaker_ids
    # reload model if language changed
    if language != model.language.split('_')[0]:
        if language == "MS":
            model = TTS(language=language, device=device, config_path=config_path, ckpt_path=ckpt_path)
        else:
            model = TTS(language=language, device=device)
        speaker_ids = model.hps.data.spk2id

    print(f"*****Final Language: {language}*****")
    print(f"*****Final Voice: {voice}*****")
    # generate speech & save to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{request.response_format}") as tmp:
        output_path = tmp.name
        model.tts_to_file(request.input, speaker_id=speaker_ids[voice], output_path=output_path, speed=request.speed, split=True)
    
    def generate():
        with open(output_path, mode="rb") as audio_file:
            yield from audio_file

    return StreamingResponse(content=generate(), media_type=media_type)


if __name__ == "__main__":
    uvicorn.run(app)
