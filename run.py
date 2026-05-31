"""
Video Session Manager
Reuses existing functions from transformation.py, read_chunks.py, query_engine.py
Adds: Upload → Process → Query → Cleanup workflow
"""

import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_NO_TORCHVISION"] = "1"

import sys
import subprocess
import uuid
from pathlib import Path
from dotenv import load_dotenv
from qdrant_client.models import Distance, VectorParams, PointStruct
from typing import Dict
# Import our existing modules
from process_video import extract_audio
from transformation import transcribe_with_groq, save_transcription
from read_chunks import (
    chunk_transcript, 
    create_embedding, 
    qdrant_client,
    # COLLECTION_NAME
)
from read_chunks import ensure_collection, upsert_chunks
from query_engine import print_answer, ask_question

import logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger(__name__)
load_dotenv()



class VideoSession:
    """
    Manages a single video session:
    - Upload & process video
    - Create temporary collection
    - Answer questions
    - Cleanup on exit
    """
    
    def __init__(self, video_path: str):
        self.video_path = video_path
        self.video_name = Path(video_path).stem
        self.session_id = str(uuid.uuid4())[:8]  # Short ID
        self.collection_name = f"session_{self.session_id}"
        
        # Create working directory
        self.work_dir = Path("temp_session") / self.session_id
        self.work_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Session started: {self.session_id}")
        print(f"Video: {self.video_name}\n")
    
    def extract_audio(self) -> str:
        """Extract audio from video"""
        print("Step 1/4: Extracting audio...")
        
        audio_path = self.work_dir / "audio.mp3"
        
        if audio_path.exists():
            print("  ✓ Audio already exists")
            return str(audio_path)
        
        try:
            extract_audio(
            self.video_path,
            str(audio_path)
            )

            print(f"Audio extracted: {audio_path.name}")
            return str(audio_path)
        
        except Exception as e:
            print(f"✗ Audio extraction error: {e}")
            raise
    
    def process_video(self):
        """Complete processing pipeline"""
        print("PROCESSING VIDEO")
        audio_path = self.extract_audio()
        audio_size = os.path.getsize(audio_path)/(1024*1024)
        print(f"Audio size: {audio_size:.2f} MB")

        print("Step 2/4 Transcribing !")
        transcript = transcribe_with_groq(audio_path)

        if not transcript:
            raise Exception("Transcription failed")
        
        print(f" Got {len(transcript['chunks'])} segments\n")

        print("Step 3/4 Chunking...")
        chunks = chunk_transcript(
            {"chunks":transcript['chunks']},
            self.video_name
            )
        print(f"Created {len(chunks)} chunks\n")

        # print("Step 4/4: Embedding & Indexing")
        # #Creating a temporary collection as we dont want to save the user session data
        # try:

        #     qdrant_client.create_collection(
        #         collection_name=self.collection_name,
        #         vectors_config=VectorParams(size=1024, distance=Distance.COSINE)
        #     )
        #     # 👉 ADD THIS HERE
        #     qdrant_client.create_payload_index(
        #         collection_name=self.collection_name,
        #         field_name="source_name",
        #         field_schema="keyword"
        #     )

        #     qdrant_client.create_payload_index(
        #         collection_name=self.collection_name,
        #         field_name="chunk_index",
        #         field_schema="integer"
        #     )
        # except:
        #     pass #Already exist
            
        # Step 4
        print("Step 4/4: Embedding & Indexing")

        ensure_collection(self.collection_name)
        try:
        # create payload indexes (keep this)
            qdrant_client.create_payload_index(
                collection_name=self.collection_name,
                field_name="source_name",
                field_schema="keyword"
            )

            qdrant_client.create_payload_index(
                collection_name=self.collection_name,
                field_name="chunk_index",
                field_schema="integer"
            )
        except Exception as e:
            log.warning(f"Payload index source_name may already exist: {e}")
        
        upsert_chunks(chunks, self.collection_name)
        print(f"  ✓ Indexed {len(chunks)} chunks\n")
        print("READY!")
       
        print(f"Indexed Chunks: {len(chunks)}\n")

        print("--------------------------")
        print("READY TO HELP YOU OUT!")
        print("--------------------------")


    def ask(self, question: str) -> Dict:
        return ask_question(question, collection_name=self.collection_name)

    def cleanup(self):
        """Delete temporary collection and files"""
        print(f"\n Cleaning up the session {self.session_id}...")

        #Delete Qdrant (VectorDB) collection
        try:
            qdrant_client.delete_collection(self.collection_name)
            print(f"Deleted collection")
        except Exception as e:
            print(f"Collection: {e}")

        #Delete temp files
        try:
            import shutil
            shutil.rmtree(self.work_dir)
            print("Deleted temp files")
        
        except Exception as e:
            print(f"Files: {e}")
        print(" Done!\n")


def interactive_session(video_path: str):
    """Run interactive Q&A"""
    session = VideoSession(video_path)
    
    try:
        # Process video
        session.process_video()
        
        # Q&A loop
        print("Type questions (or 'quit' to exit)\n")
        
        while True:
            try:
                question = input("You: ").strip()
                
                if not question:
                    continue
                
                if question.lower() in ['quit', 'exit', 'q']:
                    break
                
                # Ask and display
                result = session.ask(question)
                print_answer(result)
                print()
            
            except KeyboardInterrupt:
                print("\n\nInterrupted")
                break
    
    finally:
        session.cleanup()


def main():
    """Main entry point"""
    print(f"\n{'-'*30}")
    print(f"🎥 VIDEO Q&A SYSTEM")
    print('\n')
    
    # Get video path
    if len(sys.argv) > 1:
        video_path = sys.argv[1]
    else:
        video_path = input("Video path: ").strip()
    
    # Validate
    if not os.path.exists(video_path):
        print(f" File not found ❌: {video_path}")
        sys.exit(1)
    
    # Run session
    interactive_session(video_path)
    
    print(f"{'-'*30}")
    print(f"✅ SESSION COMPLETE")


if __name__ == "__main__":
    main()