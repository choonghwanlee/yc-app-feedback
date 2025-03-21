import pandas as pd 
from openai import OpenAI
from dotenv import load_dotenv
import os
import re
from datasets import Dataset, DatasetDict
from llm_judge.judge_prompts import clarity_system_prompt, team_market_fit_system_prompt, traction_validation_system_prompt

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def extract_score(text_response):
    # Regex pattern to find numeric scores enclosed in double square brackets
    pattern = r'\[\[(\d+)\]\]'
    
    # Search for the pattern in the text
    match = re.search(pattern, text_response)
    
    if match:
        # Return the numeric score as an integer
        return int(match.group(1))
    else:
        # Return the average score: 3
        return 3

def score_clarity(example):
    response = client.responses.create(
        model="gpt-4o-mini",
        input=[
            {
                "role": "system",
                "content": [
                    {"type":"input_text",
                     "text": clarity_system_prompt
                    }
                ]
            },
            {
                "role": "user",
                "content": [
                    {"type":"input_text",
                     "text": f"Score the clarity of this transcript: {example['transcript']}"
                    }
                ]
            }
        ],
        text={
            "format": {
                "type": "text"
            }
        },
        reasoning={},
        tools=[],
        temperature=0, 
        max_output_tokens=1000,
        top_p=1,
    )
    rationale = response.output_text
    example['clarity_score'] = extract_score(rationale)
    example['clarity_score_rationale'] = rationale
    return example

def score_team_market_fit(example):
    response = client.responses.create(
        model="gpt-4o-mini",
        input=[
            {
                "role": "system",
                "content": [
                    {"type":"input_text",
                     "text": team_market_fit_system_prompt
                    }
                ]
            },
            {
                "role": "user",
                "content": [
                    {"type":"input_text",
                     "text": f"Score the team market fit demonstrated in this transcript: {example['transcript']}"
                    }
                ]
            }
        ],
        text={
            "format": {
                "type": "text"
            }
        },
        reasoning={},
        tools=[],
        temperature=0, 
        max_output_tokens=1000,
        top_p=1,
    )
    rationale = response.output_text
    example['team_market_fit_rationale'] = rationale
    example['team_market_fit_score'] = extract_score(rationale)
    return example


def score_traction_validation(example):
    response = client.responses.create(
        model="gpt-4o-mini",
        input=[
            {
                "role": "system",
                "content": [
                    {"type":"input_text",
                     "text": traction_validation_system_prompt
                    }
                ]
            },
            {
                "role": "user",
                "content": [
                    {"type":"input_text",
                     "text": f"Score the traction and validation demonstrated in this transcript: {example['transcript']}"
                    }
                ]
            }
        ],
        text={
            "format": {
                "type": "text"
            }
        },
        reasoning={},
        tools=[],
        temperature=0, 
        max_output_tokens=1000,
        top_p=1,
    )
    rationale = response.output_text
    example['traction_validation_rationale'] = rationale
    example['traction_validation_score'] = extract_score(rationale)
    return example



if __name__ == "__main__":
    transcripts = pd.read_csv('../data/youtube_links_with_transcripts.csv')

    hf_transcripts = Dataset.from_pandas(transcripts) ## convert to HF dataset to parallelize mapping

    print("Starting clarity score calculation...")
    hf_transcripts = hf_transcripts.map(score_clarity, num_proc=6)

    print("Starting team-market fit score calculation...")
    hf_transcripts = hf_transcripts.map(score_team_market_fit, num_proc=6)

    print("Starting traction/validation score calculation...")
    hf_transcripts = hf_transcripts.map(score_traction_validation, num_proc=6)

    print("Scoring complete. Uploading HF dataset...")

    try:
        print("Uploading dataset to Hugging Face Hub...")
        hf_transcripts.push_to_hub("jasonhwan/yc-startup-pitches-with-scores")
        print("Uploaded Successfully!")
    except Exception as e:
        print(f"Failed to upload dataset: {e}")
