# converts the videos to mp3
# import whisper
import os
import subprocess


def extract_audio(video_path, output_path):
    # tutorial_number = file.split(".")[0]
  
    # file_name = os.path.splitext(file)[0] #The [0] simply grabs the first element — the filename without the extension
    # as os.path.splitext(file) returns a tuple (filename, extension)
    # print(file_name)
    subprocess.run(["ffmpeg", 
                    "-y",
                    "-i", video_path,
                    "-vn", # tells ffmpeg ignore video streams
                    "-ac", "1",
                    "-ar", "16000",
                    "-b:a", "32k",
                    # f"audios/{file_name}.mp3"],
                    output_path
                ],
                    check=True
                    )
    # audio_path=f"audios/{file_name}.mp3"
    # subprocess.run(["ffprobe", audio_path])
    # audio_size = os.path.getsize(audio_path) / (1024 * 1024)
    # print(f"\n{file_name}")
    # print(f"Audio Size: {audio_size:.2f} MB")
    audio_size = os.path.getsize(output_path)/(1024*1024)
    print(f"Audio Size: {audio_size:.2f} MB")

    return output_path

