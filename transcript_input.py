import os
import subprocess
import whisper

def download_youtube_video(url, output_file="pitch_video.mp4"):
    """
    Download YouTube video using yt-dlp (force mp4 format).
    """
    command = [
        "yt-dlp",
        "-f", "mp4",
        "-o", output_file,
        url
    ]
    subprocess.run(command, check=True)
    print(f" Video downloaded: {output_file}")
    return output_file

def transcribe_with_whisper(video_file, output_file="pitch_transcript.txt"):
    """
    Transcribe audio using Whisper and save to a text file.
    """
    model = whisper.load_model("base")
    result = model.transcribe(video_file)

    with open(output_file, "w", encoding="utf-8") as f:
        f.write(result["text"])

    print(f" Transcription complete. Saved to: {output_file}")
    return result["text"]

def process_youtube_video():
    """
    Download and transcribe YouTube video.
    """
    youtube_url = input("Enter YouTube video URL: ").strip()
    video_file = download_youtube_video(youtube_url)
    transcript = transcribe_with_whisper(video_file)
    print("\nTranscript preview (first 200 chars):")
    print(transcript[:200])

def process_local_file():
    """
    Transcribe already downloaded local file.
    """
    file_path = input("Enter local file path: ").strip()
    if not os.path.exists(file_path):
        print("File not found.")
        return
    transcript = transcribe_with_whisper(file_path, "local_file_transcript.txt")
    print("\nTranscript preview (first 200 chars):")
    print(transcript[:200])

if __name__ == "__main__":
    print("Choose input type:")
    print("1️. YouTube video link")
    print("2️. Local file")
    choice = input("Enter 1 or 2: ").strip()

    if choice == "1":
        process_youtube_video()
    elif choice == "2":
        process_local_file()
    else:
        print("Invalid choice.")
