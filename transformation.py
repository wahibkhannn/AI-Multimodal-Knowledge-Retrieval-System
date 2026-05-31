# transformation.py
"""
Audio → Transcript transformation using Groq (online) or Whisper (local)
"""

import json
import os
import time
from typing import Dict

from dotenv import load_dotenv
from groq import Groq

# =========================
# CONFIG
# =========================
USE_GROQ = True   # True = Groq | False = Local Whisper

AUDIO_FOLDER = "audios"
OUTPUT_FOLDER = "transcripts"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)
load_dotenv()

# =========================
# CLIENT INITIALIZATION
# =========================
groq_client = None
model = None

if USE_GROQ:
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    if not GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY not found in .env file")

    groq_client = Groq(api_key=GROQ_API_KEY)
    print("--- Using Groq Whisper-large-v3 (Fast Mode) ---")

else:
    import whisper  # imported ONLY if needed
    model = whisper.load_model("large-v2")
    print("--- Using Local Whisper-large-v2 (Offline Mode) ---")

# =========================
# HELPERS
# =========================
def save_transcription(
    audio_name: str,
    chunks: list,
    full_text: str,
    duration: float = None,
    language: str = None
):
    output_path = os.path.join(
        OUTPUT_FOLDER, audio_name.replace(".mp3", ".json")
    )

    output_data = {
        "audio_file": audio_name,
        "duration": duration,
        "language": language,
        "chunks": chunks,
        "full_text": full_text,
        "chunk_count": len(chunks),
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=4)

    print(f"-- Saved transcript: {output_path}")

# =========================
# GROQ TRANSCRIPTION
# =========================
def transcribe_with_groq(audio_path: str) -> Dict | None:
    import traceback

    audio_size = os.path.getsize(audio_path) / (1024 * 1024)
    print(f"Audio size: {audio_size:.2f} MB")

    print("Transcribing with Groq...")
    start_time = time.time()

    try:
        with open(audio_path, "rb") as f:
            response = groq_client.audio.translations.create(
                # file=(os.path.basename(audio_path), f.read()),
                file=f,
                model="whisper-large-v3",
                response_format="verbose_json",
                temperature=0.0,
            )

        chunks = [
            {
                "start": seg["start"],
                "end": seg["end"],
                "text": seg["text"],
            }
            for seg in response.segments
        ]

        elapsed = time.time() - start_time
        print(f"Done in {elapsed:.2f}s")

        return {
            "chunks": chunks,
            "full_text": response.text,
            "duration": response.duration,
            "language": response.language,
        }

    
    except Exception as e:
        print("\n========== FULL ERROR ==========")
        print(type(e))
        print(e)
        traceback.print_exc()
        print("================================\n")

# =========================
# LOCAL WHISPER TRANSCRIPTION
# =========================
def transcribe_with_local(audio_path: str) -> Dict | None:
    print("Transcribing locally...")
    start_time = time.time()

    try:
        result = model.transcribe(
            audio=audio_path,
            task="translate",
            word_timestamps=False,
        )

        chunks = [
            {
                "start": seg["start"],
                "end": seg["end"],
                "text": seg["text"],
            }
            for seg in result["segments"]
        ]

        elapsed = (time.time() - start_time) / 60
        print(f"Done in {elapsed:.2f} min")

        return {
            "chunks": chunks,
            "full_text": result["text"],
            "duration": None,
            "language": result.get("language"),
        }

    except Exception as e:
        print(f"[Local Whisper Error] {e}")
        return None

# =========================
# PROCESS AUDIO FILES
# =========================
def process_audio_file(audio_folder: str, file_pattern: str = None):
    audio_files = [
        f for f in os.listdir(audio_folder) if f.endswith(".mp3")
    ]

    if file_pattern:
        audio_files = [f for f in audio_files if f.startswith(file_pattern)]

    if not audio_files:
        print("No audio files found.")
        return

    print(f"Found {len(audio_files)} audio files")

    success, fail = 0, 0

    for idx, audio_file in enumerate(audio_files, 1):
        print(f"\n[{idx}/{len(audio_files)}] {audio_file}")
        audio_path = os.path.join(audio_folder, audio_file)
        
        result = (
            transcribe_with_groq(audio_path)
            if USE_GROQ
            else transcribe_with_local(audio_path)
        )

        if result:
            save_transcription(
                audio_name=audio_file,
                chunks=result["chunks"],
                full_text=result["full_text"],
                duration=result.get("duration"),
                language=result.get("language"),
            )
            success += 1
        else:
            print(f"Failed: {audio_file}")
            fail += 1

    print(f"\nCompleted → Success: {success}, Failed: {fail}")

# =========================
# MAIN
# =========================
if __name__ == "__main__":
    process_audio_file(AUDIO_FOLDER)
