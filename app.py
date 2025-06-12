import os
import argparse
import tempfile
import uvicorn
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from dotenv import load_dotenv
from contextlib import asynccontextmanager
from melo.api import TTS
from lingua import Language, LanguageDetectorBuilder

load_dotenv()
TTS_LANGUAGE = os.getenv("TTS_LANGUAGE", "auto") # EN -> English, ES -> Spanish, FR -> French, JP -> Japanese, KR -> Korean, ZH_MIX_EN -> Chinese, MS -> Malay
TTS_VOICE = os.getenv("TTS_VOICE", "auto") # EN-US, EN-BR, EN-INDIA, EN-AU, EN-Default, ES, FR, JP, KR, ZH, husein-chatbot, shafiqah-idayu-chatbot, anwar-ibrahim
TTS_RESPONSE_FORMAT = os.getenv("TTS_RESPONSE_FORMAT", "mp3") # mp3, opus, aac, flac, wav
TTS_SPEED = float(os.getenv("TTS_SPEED", 1.0)) # 0.25 - 4.0

device = "auto"
supported_languages = [Language.ENGLISH, Language.SPANISH, Language.FRENCH, Language.JAPANESE, Language.KOREAN, Language.CHINESE, Language.MALAY]
language_detector = LanguageDetectorBuilder.from_languages(*supported_languages).build()
languages_voices = {
    'EN': ['EN-US', 'EN-BR', 'EN-INDIA', 'EN-AU', 'EN-Default'], 
    'ES': ['ES'],
    'FR': ['FR'],
    'JP': ['JP'], 
    'KR': ['KR'], 
    'ZH': ['ZH'], 
    'MS': ['husein-chatbot', 'shafiqah-idayu-chatbot', 'anwar-ibrahim']
}

# get TTS model for MS language
from huggingface_hub import hf_hub_download
ckpt_path = hf_hub_download(repo_id='mesolitica/MeloTTS-MS', filename='model.pth')
config_path = hf_hub_download(repo_id='mesolitica/MeloTTS-MS', filename='config.json')

@asynccontextmanager
async def lifespan(app: FastAPI):
    # load TTS models
    global models
    models = {
        'EN': TTS(language='EN', device=device),
        'ES': TTS(language='ES', device=device),
        'FR': TTS(language='FR', device=device),
        'ZH': TTS(language='ZH', device=device),
        'JP': TTS(language='JP', device=device),
        'KR': TTS(language='KR', device=device),
        'MS': TTS(language='MS', device=device, config_path=config_path, ckpt_path=ckpt_path),
    }
    yield
    # clean up TTS model & release resources
    del models

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

    # auto detect language if language is auto
    if TTS_LANGUAGE == "auto":
        language = language_detector.detect_language_of(request.input)
        language = "EN" if language is None else language.iso_code_639_1.name # get language code
    else:
        language = TTS_LANGUAGE

    # match the language codes of lingua-py to MeloTTS
    if language == "JA":
        language = "JP"
    elif language == "KO":
        language = "KR"
    
    # set the voice as requested if the voice match with language,
    # otherwise set the voice based on the language
    if request.voice in languages_voices[language]:
        voice = request.voice
    else:
        voice = language
        if language == "EN":
            voice = "EN-Default"
        elif language == "MS":
            voice = "shafiqah-idayu-chatbot"

    global models

    print(f"*****Final Language: {language}*****")
    print(f"*****Final Voice: {voice}*****")
    # generate speech & save to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{request.response_format}") as tmp:
        output_path = tmp.name
        speaker_ids = models[language].hps.data.spk2id
        models[language].tts_to_file(request.input, speaker_id=speaker_ids[voice], output_path=output_path, speed=request.speed, split=True)
    
    def generate():
        with open(output_path, mode="rb") as audio_file:
            yield from audio_file

    return StreamingResponse(content=generate(), media_type=media_type)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
