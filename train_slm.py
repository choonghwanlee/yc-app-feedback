from trl import SFTTrainer
from transformers import TrainingArguments, DataCollatorForSeq2Seq
from unsloth import FastLanguageModel, is_bfloat16_supported
from unsloth.chat_templates import get_chat_template, train_on_responses_only
from datasets import Dataset, load_dataset
from llm_judge.judge_prompts import clarity_system_prompt, team_market_fit_system_prompt, traction_validation_system_prompt
from llm_judge.label_transcripts import extract_score
from sklearn.metrics import cohen_kappa_score


# Function to expand each row
def expand_row(example):
    rationale_types = ["clarity_score", "team_market_fit_score", "traction_validation_score"]
    
    expanded_rows = {
        "title": [],
        "link": [],
        "transcript": [],
        "rationale_type": [],
        "rationale": [],
        "score": []
    }

    for rtype in rationale_types:
        expanded_rows["title"].append(example["title"][0])
        expanded_rows["link"].append(example["link"][0])
        expanded_rows["transcript"].append(example["transcript"][0])
        expanded_rows["rationale_type"].append(rtype)
        expanded_rows["rationale"].append(example[f"{rtype}_rationale"][0])
        expanded_rows["score"].append(example[rtype][0])
    
    return expanded_rows

def prepare_model(model_name = "unsloth/Llama-3.2-3B-Instruct-bnb-4bit", max_seq_length = 4096):

    max_seq_length = max_seq_length
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        load_in_4bit=True,
        dtype=None,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        lora_alpha=16,
        lora_dropout=0,
        target_modules=["q_proj", "k_proj", "v_proj", "up_proj", "down_proj", "o_proj", "gate_proj"],
        use_rslora=True,
        use_gradient_checkpointing="unsloth",
        random_state = 42,
        loftq_config = None,
    )


    tokenizer = get_chat_template(
        tokenizer,
        chat_template = "llama-3.2",
    )

    return model, tokenizer

def formatting_prompts_func(examples, model, tokenizer, train=True): 
  batch_size = len(examples['transcript'])
  conversations = []

  for i in range(batch_size):
      if examples['rationale_type'][i] == 'clarity_score':
          system_prompt = clarity_system_prompt
          user_prompt = f"Score the clarity of this transcript: {examples['transcript'][i]}"
      elif examples['rationale_type'][i] == 'team_market_fit_score':
          system_prompt = team_market_fit_system_prompt
          user_prompt = f"Score the team market fit demonstrated in this transcript: {examples['transcript'][i]}"
      else:
          system_prompt = traction_validation_system_prompt
          user_prompt = f"Score the traction and validation demonstrated in this transcript: {examples['transcript'][i]}"

      conversation = [
          {'role': 'system', 'content': system_prompt},
          {'role': 'user', 'content': user_prompt},
          *([{'role': 'assistant', 'content': examples['rationale'][i]}] if train else [])
      ]
      conversations.append(
        tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=False if train else True)
      )
      if not train:
            inputs = tokenizer.apply_chat_template(
            conversation,
            tokenize = True,
            add_generation_prompt = True, # Must add for generation
            return_tensors = "pt",
        ).to("cuda")
      outputs = model.generate(input_ids = inputs, max_new_tokens = 1024, use_cache = True,
                                temperature = 0.1, min_p = 0.1)
      generated_tokens = outputs[:, inputs.shape[1]:]
      generated_text = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
      predicted = [extract_score(text) for text in generated_text]
      return {'predicted_score': predicted}
  return {'text': conversations}

def train_model(model, tokenizer, train_dataset, max_seq_length = 4096):
    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = train_dataset,
        dataset_text_field = "text",
        max_seq_length = max_seq_length,
        data_collator = DataCollatorForSeq2Seq(tokenizer = tokenizer),
        dataset_num_proc = 2,
        packing = False, # Can make training 5x faster for short sequences.
        args = TrainingArguments(
            per_device_train_batch_size = 2,
            gradient_accumulation_steps = 4,
            warmup_steps = 5,
            num_train_epochs = 5, # Set this for 1 full training run.
            max_steps = 60,
            learning_rate = 2e-4,
            fp16 = not is_bfloat16_supported(),
            bf16 = is_bfloat16_supported(),
            logging_steps = 1,
            optim = "adamw_8bit",
            weight_decay = 0.01,
            lr_scheduler_type = "linear",
            seed = 3407,
            output_dir = "outputs",
            report_to = "none", # Use this for WandB etc
        ),
    )


    trainer = train_on_responses_only(
        trainer,
        instruction_part = "<|start_header_id|>user<|end_header_id|>\n\n",
        response_part = "<|start_header_id|>assistant<|end_header_id|>\n\n",
    )

    trainer_stats = trainer.train()

    model.push_to_hub("jasonhwan/yc-acceptance-predictor-distilled") # Online saving
    tokenizer.push_to_hub("jasonhwan/yc-acceptance-prediction-distilled") # Online saving

    print('Done training. Model pushed to Hugging Face Hub')

    return trainer_stats

if __name__ == '__main__':
    dataset = load_dataset("jasonhwan/yc-startup-pitches-with-scores", split='train')
    # Apply the function using .map with batched=False
    expanded_dataset = dataset.map(expand_row, remove_columns=dataset.column_names, batched=True, batch_size=1)
    split_dataset = expanded_dataset.train_test_split(test_size=0.2)

    train_dataset = split_dataset["train"]
    eval_dataset = split_dataset["test"]
    model, tokenizer = prepare_model()
    train_dataset = train_dataset.map(formatting_prompts_func, fn_kwargs={'tokenizer': tokenizer}, batched = True,)
    train_model(model, tokenizer, train_dataset)
    model.push_to_hub("jasonhwan/yc-acceptance-predictor-distilled") # Online saving
    tokenizer.push_to_hub("jasonhwan/yc-acceptance-prediction-distilled") # Online saving
    
    FastLanguageModel.for_inference(model) # Enable native 2x faster inference
    eval_dataset = eval_dataset.map(formatting_prompts_func, fn_kwargs={'model': model, 'tokenizer': tokenizer, 'train': False}, batched = True,)
    print("QWR score: ", cohen_kappa_score(eval_dataset['predicted_score'], eval_dataset['score'], weights='quadratic'))


    
