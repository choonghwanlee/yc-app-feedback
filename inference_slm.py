
import torch
from unsloth import FastLanguageModel
from llm_judge.judge_prompts import clarity_system_prompt, team_market_fit_system_prompt, traction_validation_system_prompt
from llm_judge.label_transcripts import extract_score

model, tokenizer = FastLanguageModel.from_pretrained(
    "jasonhwan/yc-acceptance-prediction-distilled",
    max_seq_length = 4096,
    load_in_4bit=True,
    dtype=None,
)

def forward(transcript, prompt_type = 'clarity'):
    '''
    Forward pass for the trained SLM model

    Inputs:
        transcript (str): a single transcript to score
        prompt_type (str): one of 3 categories to score with. can be clarity, team_market_fit, or traction_validation. defaults to clarity
    Outputs:
        generated_text (str): the chain of thought, including the predicted score at the end
        predicted (int): the predicted score
    '''
    device = 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'
    FastLanguageModel.for_inference(model)
    if prompt_type == 'clarity':
        system_prompt = clarity_system_prompt
        user_prompt = f"Score the clarity of this transcript: {transcript}"
    elif prompt_type == 'team_market_fit':
        system_prompt = team_market_fit_system_prompt
        user_prompt = f"Score the team market fit demonstrated in this transcript: {transcript}"
    else:
        system_prompt = traction_validation_system_prompt
        user_prompt = f"Score the traction and validation demonstrated in this transcript: {transcript}"

    conversation = [
        {'role': 'system', 'content': system_prompt},
        {'role': 'user', 'content': user_prompt},
    ]
    inputs = tokenizer.apply_chat_template(
        conversation,
        tokenize = True,
        add_generation_prompt = True, # Must add for generation
        return_tensors = "pt",
    ).to(device)
    outputs = model.generate(input_ids = inputs, max_new_tokens = 1024, use_cache = True,
                            temperature = 0.1, min_p = 0.1)
    generated_tokens = outputs[:, inputs.shape[1]:]
    generated_text = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
    predicted = extract_score(generated_text)
    return generated_text, predicted