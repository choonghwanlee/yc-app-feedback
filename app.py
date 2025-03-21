import streamlit as st
import torch
from transformers import AutoTokenizer
import whisper
import subprocess
import os
import pandas as pd
from inference_slm import model, tokenizer, forward  # Import model

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


# Streamlit App UI
st.set_page_config(page_title="Pitch Evaluation App", layout="wide")
st.title("🚀 Pitch Evaluation")

option = st.radio("Choose Input Method", ("YouTube URL", "Upload File"), horizontal=True)

if option == "YouTube URL":
    url = st.text_input("🎥 Enter YouTube URL")
    if st.button("Transcribe and Grade", use_container_width=True):
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
    clarity_text, clarity = forward(transcript, prompt_type='clarity')
    team_text, team = forward(transcript, prompt_type='team_market_fit')
    traction_text, traction = forward(transcript, prompt_type='traction_validation')
    
    if None not in (clarity, team, traction):
        # Create a DataFrame for the scoring table
        categories = ["Clarity & Conciseness", "Team-Market Fit", "Traction / Validation"]
        scores = [clarity, team, traction]
        explanations = [clarity_text, team_text, traction_text]
        df = pd.DataFrame({"Category": categories, "Score (1-5)": scores, "Explanation": explanations})
        
        st.write("## 📊 Evaluation Results")
        st.table(df)

        if ((clarity + team + traction)/3) >= 3.5:
            st.write("## 🎉 Congrats! You have a high possibility to be accepted")
        else:
            st.write("## 🙌 Need More Practice, but don't give up!")

