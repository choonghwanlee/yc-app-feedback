import streamlit as st
import torch
from transformers import AutoTokenizer
import subprocess
import whisper
import os
import pandas as pd
from inference_slm import forward  # Import model

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
        devices = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
        model = whisper.load_model("base", device= devices)
        result = model.transcribe(video_file)
        return result["text"]
    except Exception as e:
        st.error("❌ An error occurred during transcription.")
        return ""


# Streamlit App UI
st.set_page_config(page_title="Pitch Evaluation App", layout="wide")
st.title("🚀 Pitch Evaluation")

option = st.radio("Choose Input Method", ("Upload File"), horizontal=True)

if option == "Upload File":
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
        descriptions = [
            "Extremely clear, direct, and easy to follow; no fluff, just essential details." if clarity == 5 else "Mostly clear, with only minor unnecessary details." if clarity == 4 else "Somewhat clear but includes extra details or minor distractions." if clarity == 3 else "Lacks clarity; hard to follow; too much fluff or filler." if clarity == 2 else "Unclear, rambling, and difficult to understand.",
            "Founders have highly relevant skills & experience to execute this successfully." if team == 5 else "Founders have good experience but may lack some key skills." if team == 4 else "Some relevant experience but gaps in expertise." if team == 3 else "Limited relevant experience; execution ability is questionable." if team == 2 else "No clear expertise in this space; team seems unqualified.",
            "Strong proof of demand (users, revenue, engagement, partnerships, etc.)." if traction == 5 else "Good early validation with promising signs of demand." if traction == 4 else "Some traction but not yet convincing." if traction == 3 else "Weak or vague traction, with little evidence of demand." if traction == 2 else "No validation or proof that people want this."
        ]
        df = pd.DataFrame({"Category": categories, "Score (1-5)": scores, "Description": descriptions})
        
        st.write("## 📊 Evaluation Results")
        st.table(df)
        with st.expander("See Explanations 👀"):
            st.write(f"""
                ### **Clarity Score Explanation:**  \n
                {clarity_text}
            """)
            st.write(f"""
                ### **Team-Market Fit Explanation:**  \n
                {team_text}
            """)
            st.write(f"""
                ### **Traction & Validation Explanation:**  \n
                {traction_text}
            """)
        if ((clarity + team + traction)/3) >= 3.5:
            st.write("## 🎉 Congrats! You have a high possibility to be accepted")
        else:
            st.write("## 🙌 Need More Practice, but don't give up!")

