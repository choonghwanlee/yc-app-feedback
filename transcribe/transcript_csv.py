import os
import csv
import subprocess
import whisper
import re

# Ensure the directory exists
os.makedirs("videos", exist_ok=True)

def sanitize_filename(name):
    """
    Remove special characters and spaces from the filename to ensure compatibility
    """
    return re.sub(r'[^a-zA-Z0-9_\-]', '_', name)

def check_ffmpeg():
    """
    Check if ffmpeg is available
    """
    try:
        subprocess.run(["ffmpeg", "-version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    except FileNotFoundError:
        raise RuntimeError("ffmpeg not found. Please install ffmpeg first!")

def download_youtube_video(url):
    """
    Download video to a fixed path: videos/temp_video.mp4
    """
    output_path = "videos/temp_video.mp4"
    command = [
        "yt-dlp",
        "-f", "mp4",
        "-o", output_path,
        url
    ]
    subprocess.run(command, check=True)
    return output_path  # Return the fixed file path

def find_downloaded_file(folder, video_id):
    """
    Find the matching video file in the 'videos' folder (not really needed if you use fixed names)
    """
    for file in os.listdir(folder):
        if video_id in file and file.endswith(".mp4"):
            return os.path.join(folder, file)

    raise FileNotFoundError(f"No mp4 file found for {video_id} in {folder}")

def transcribe_with_whisper(video_file):
    """
    Use Whisper to transcribe audio and return the text
    """
    model = whisper.load_model("base")
    result = model.transcribe(video_file)
    return result["text"]

def process_csv(input_csv, output_csv):
    """
    Process each video: download, transcribe, delete, and save results to CSV (with resume support)
    """
    os.makedirs("videos", exist_ok=True)

    processed_links = set()
    if os.path.exists(output_csv):
        with open(output_csv, newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                processed_links.add(row['link'])

    fieldnames = ['title', 'link', 'transcript']
    with open(output_csv, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if f.tell() == 0:
            writer.writeheader()

        with open(input_csv, newline='', encoding='utf-8') as infile:
            reader = csv.DictReader(infile)
            for row in reader:
                url = row['link']
                if url in processed_links:
                    print(f"Already processed, skipping: {url}")
                    continue

                video_id = sanitize_filename(row['title'])

                try:
                    print(f"\nProcessing: {video_id} - {url}")

                    video_file = download_youtube_video(url)
                    transcript = transcribe_with_whisper(video_file)
                    os.remove(video_file)

                    row['transcript'] = transcript
                    writer.writerow(row)

                except Exception as e:
                    print(f" Failed to process: {video_id} - {e}")
                    row['transcript'] = f"ERROR: {e}"
                    writer.writerow(row)

if __name__ == "__main__":
    input_csv = "./youtube_search_results.csv"
    output_csv = "youtube_links_with_transcripts.csv"

    check_ffmpeg()

    process_csv(input_csv, output_csv)
