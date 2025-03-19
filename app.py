import streamlit as st
import torch
from transformers import AutoTokenizer
import whisper
import subprocess
import os
import pandas as pd
from dl import PitchEvaluationModel  # Import model

def download_youtube_video(url, output_file="pitch_video.mp4"):
    """Download YouTube video using yt-dlp."""
    if "youtube.com" not in url and "youtu.be" not in url:
        st.error("❌ Invalid URL! Please enter a valid YouTube link.")
        return None
    try:
        command = ["yt-dlp", "-f", "mp4", "-o", output_file, url]
        subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return output_file
    except subprocess.CalledProcessError:
        st.error("❌ Failed to download the video. Please check the URL and try again.")
        return None

def transcribe_video(video_file):
    """Transcribe video using Whisper."""
    try:
        model = whisper.load_model("base")
        result = model.transcribe(video_file)
        return result["text"]
    except Exception as e:
        st.error("❌ An error occurred during transcription.")
        return ""

def load_model():
    """Load the trained model."""
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = PitchEvaluationModel("bert-base-uncased").to(device)
        model.load_state_dict(torch.load("best_pitch_model.pt", map_location=device))
        model.eval()
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        return model, tokenizer, device
    except Exception as e:
        st.error("❌ Failed to load the model.")
        return None, None, None

def evaluate_pitch(transcript, model, tokenizer, device):
    """Evaluate transcript using the trained model."""
    try:
        inputs = tokenizer(transcript, return_tensors="pt", truncation=True, padding="max_length", max_length=512)
        input_ids, attention_mask = inputs["input_ids"].to(device), inputs["attention_mask"].to(device)
        with torch.no_grad():
            clarity, team, traction = model(input_ids, attention_mask)
        return torch.argmax(clarity).item() + 1, torch.argmax(team).item() + 1, torch.argmax(traction).item() + 1
    except Exception as e:
        st.error("❌ Error in evaluation process.")
        return None, None, None

# Streamlit App UI
st.set_page_config(page_title="Pitch Evaluation App", layout="wide")
st.title("🚀 Pitch Evaluation")

option = st.radio("Choose Input Method", ("YouTube URL", "Upload File"), horizontal=True)

if option == "YouTube URL":
    url = st.text_input("🎥 Enter YouTube URL")
    if st.button("Download and Transcribe", use_container_width=True):
        video_file = download_youtube_video(url)
        if video_file:
            transcript = transcribe_video(video_file)
            st.text_area("📜 Transcript", transcript, height=200)
elif option == "Upload File":
    uploaded_file = st.file_uploader("📂 Upload Video", type=["mp4"], help="Upload a video file for transcription and evaluation.")
    if uploaded_file is not None:
        if uploaded_file.type != "video/mp4":
            st.error("❌ Invalid file format! Please upload an MP4 file.")
        else:
            with open("uploaded_video.mp4", "wb") as f:
                f.write(uploaded_file.getbuffer())
            transcript = transcribe_video("uploaded_video.mp4")
            st.text_area("📜 Transcript", transcript, height=200)

if 'transcript' in locals() and transcript:
    model, tokenizer, device = load_model()
    if model is not None:
        clarity, team, traction = evaluate_pitch(transcript, model, tokenizer, device)
        if None not in (clarity, team, traction):
            # Create a DataFrame for the scoring table
            categories = ["Clarity & Conciseness", "Team-Market Fit", "Traction / Validation"]
            scores = [clarity, team, traction]
            descriptions = [
                "Extremely clear, direct, and easy to follow;no fluff, just essential details." if clarity == 5 else "Mostly clear, with only minor unnecessary details." if clarity == 4 else "Somewhat clear but includes extra details or minor distractions." if clarity == 3 else "Lacks clarity; hard to follow; too much fluff or filler." if clarity == 2 else "Unclear, rambling, and difficult to understand.",
                "Founders have highly relevant skills & experience to execute this successfully." if team == 5 else "Founders have good experience but may lack some key skills." if team == 4 else "Some relevant experience but gaps in expertise." if team == 3 else "Limited relevant experience; execution ability is questionable." if team == 2 else "No clear expertise in this space; team seems unqualified.",
                "Strong proof of demand (users, revenue, engagement, partnerships, etc.)." if traction == 5 else "Good early validation with promising signs of demand." if traction == 4 else "Some traction but not yet convincing." if traction == 3 else "Weak or vague traction, with little evidence of demand." if traction == 2 else "No validation or proof that people want this."
            ]
            df = pd.DataFrame({"Category": categories, "Score (1-5)": scores, "Evaluation": descriptions})
            
            st.write("## 📊 Evaluation Results")
            st.table(df)

        if ((clarity + team + traction)/3) >=3.5:
            st.write("## 🎉 Congrats! You have a high possibility to be accepted")
        else:
            st.write("## 🙌 Need More Practice, but don't give up!")

