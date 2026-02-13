from fastapi import APIRouter, UploadFile, File
from app.speech.stt import speech_to_text

router= APIRouter()

@router.post("/stt")
async def transcribe(file: UploadFile=File(...)):
    audio_bytes=await file.read()
    text = speech_to_text(audio_bytes)
    return {"text": text}