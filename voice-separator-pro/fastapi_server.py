from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import librosa
import soundfile as sf
import numpy as np
import tempfile
import os
from pydub import AudioSegment
from noisereduce import reduce_noise

# --------------------------------------------------
# APP SETUP
# --------------------------------------------------
app = FastAPI(title="Voice Separator Pro - CPU Edition")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------------------------------------------------
# CONSTANTS
# --------------------------------------------------
CLEAN_FILENAME = "clean_audio.wav"
CLEAN_PATH = os.path.join("/tmp", CLEAN_FILENAME)  # safer on Render

# --------------------------------------------------
# ROUTES
# --------------------------------------------------
@app.get("/")
async def root():
    return {"message": "Voice Separator Pro - CPU Mode ✅"}

@app.get("/health")
async def health():
    return {"status": "healthy", "mode": "CPU"}

@app.post("/api/separate")
async def separate_audio(file: UploadFile = File(...)):
    tmp_path = None

    try:
        # Save uploaded file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name

        # Load audio
        try:
            audio, sr = librosa.load(tmp_path, sr=22050, mono=True)
        except Exception:
            audio_seg = AudioSegment.from_file(tmp_path)
            samples = np.array(audio_seg.get_array_of_samples())

            if audio_seg.channels == 2:
                samples = samples.reshape((-1, 2)).mean(axis=1)

            audio = samples.astype(np.float32) / 32768.0
            sr = 22050

        duration = len(audio) / sr

        # Noise reduction
        try:
            noise_clip = audio[: int(sr * 0.5)]
            audio_clean = reduce_noise(y=audio, sr=sr, y_noise=noise_clip)
        except Exception:
            audio_clean = audio

        # Simple energy-based VAD
        hop_length = 512
        frame_length = 2048
        energy = librosa.feature.rms(
            y=audio_clean,
            frame_length=frame_length,
            hop_length=hop_length
        )[0]

        threshold = np.mean(energy) * 2.5
        speech_frames = energy > threshold

        segments = []
        i = 0
        while i < len(speech_frames):
            if speech_frames[i]:
                start = i
                while i < len(speech_frames) and speech_frames[i]:
                    i += 1
                end = i

                dur = (end - start) * hop_length / sr
                if dur > 0.2:
                    segments.append({
                        "start": round(start * hop_length / sr, 2),
                        "end": round(end * hop_length / sr, 2),
                        "duration": round(dur, 2),
                    })
            else:
                i += 1

        total_speech = sum(seg["duration"] for seg in segments)
        speech_pct = (total_speech / duration * 100) if duration > 0 else 0

        # Save cleaned audio
        sf.write(CLEAN_PATH, audio_clean, sr)

        return {
            "status": "success",
            "filename": file.filename,
            "duration": round(duration, 2),
            "speech_duration": round(total_speech, 2),
            "speech_percentage": round(min(100.0, speech_pct), 2),
            "segments": segments[:10],
            "clean_file": CLEAN_FILENAME,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)

@app.get("/download/{filename}")
async def download_file(filename: str):
    if filename != CLEAN_FILENAME or not os.path.exists(CLEAN_PATH):
        raise HTTPException(status_code=404, detail="File not found")

    return FileResponse(
        CLEAN_PATH,
        media_type="audio/wav",
        filename=CLEAN_FILENAME
    )
