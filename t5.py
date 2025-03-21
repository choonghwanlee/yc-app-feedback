import pandas as pd
import json
from datasets import Dataset
from transformers import T5ForConditionalGeneration, T5Tokenizer, Trainer, TrainingArguments
from sklearn.metrics import cohen_kappa_score
from datasets import load_dataset
import pandas as pd

dataset = load_dataset("jasonhwan/yc-startup-pitches-with-scores", split="train")
df = dataset.to_pandas()

df.to_csv("yc_startup_pitches.csv", index=False)

# Step 1: Read the CSV file and generate a JSON file
df = pd.read_csv("yc_startup_pitches.csv")
json_list = []

for idx, row in df.iterrows():
    # Construct the JSON data structure
    data_dict = {
        "id": idx,
        "transcript": row["transcript"],
        "rationales": {
            "clarity": {
                "rationale": row["clarity_score_rationale"],
                "score": row["clarity_score"]
            },
            "team_market_fit": {
                "rationale": row["team_market_fit_score_rationale"],
                "score": row["team_market_fit_score"]
            },
            "traction_validation": {
                "rationale": row["traction_validation_score_rationale"],
                "score": row["traction_validation_score"]
            }
        }
    }
    json_list.append(data_dict)

# Save as a JSON file
json_str = json.dumps(json_list, ensure_ascii=False, indent=2)
with open("output.json", "w", encoding="utf-8") as f:
    f.write(json_str)

print("JSON file generated: output.json")

# Step 2: Read the generated JSON file and prepare training data
with open("output.json", "r", encoding="utf-8") as f:
    data = json.load(f)

train_data = []
for record in data:
    transcript = record["transcript"]
    
    clarity_rationale = record["rationales"]["clarity"]["rationale"]
    clarity_score = record["rationales"]["clarity"]["score"]
    
    team_rationale = record["rationales"]["team_market_fit"]["rationale"]
    team_score = record["rationales"]["team_market_fit"]["score"]
    
    traction_rationale = record["rationales"]["traction_validation"]["rationale"]
    traction_score = record["rationales"]["traction_validation"]["score"]
    
    prompt = (
        f"Pitch: {transcript}\n"
        "Please analyze the clarity, team-market fit, and traction validation of this pitch. "
        "Provide detailed analysis and give a final score (1-5) for each dimension."
    )
    
    target = (
        f"Clarity analysis: {clarity_rationale} Final clarity score: {clarity_score}.\n"
        f"Team-market fit analysis: {team_rationale} Final team-market fit score: {team_score}.\n"
        f"Traction validation analysis: {traction_rationale} Final traction validation score: {traction_score}."
    )
    
    train_data.append({"input_text": prompt, "target_text": target})

# Convert to a HuggingFace Dataset
dataset = Dataset.from_list(train_data)

# Step 3: Data Preprocessing
from datasets import DatasetDict

# Tokenization function
def tokenize_function(examples):
    inputs = tokenizer(examples["input_text"], padding="max_length", truncation=True, max_length=512)
    targets = tokenizer(examples["target_text"], padding="max_length", truncation=True, max_length=512)
    inputs["labels"] = targets["input_ids"]
    return inputs

# Initialize the tokenizer
tokenizer = T5Tokenizer.from_pretrained("t5-small")

# Apply the tokenizer to the dataset
train_dataset = dataset.map(tokenize_function, batched=True)

# Split the dataset into training and validation sets (80% train, 20% validation)
split_dataset = train_dataset.train_test_split(test_size=0.2)
train_dataset = split_dataset["train"]
eval_dataset = split_dataset["test"]

# Step 4: Load the pre-trained model
model = T5ForConditionalGeneration.from_pretrained("t5-small")

# Step 5: Set training parameters
training_args = TrainingArguments(
    output_dir="./results",             
    num_train_epochs=3,                 
    per_device_train_batch_size=8,    
    per_device_eval_batch_size=8,    
    evaluation_strategy="epoch",       
    save_strategy="epoch",           
    logging_dir="./logs",             
    logging_steps=10,                 
    save_total_limit=2,                
    load_best_model_at_end=True,       
)

trainer = Trainer(
    model=model,                     
    args=training_args,             
    train_dataset=train_dataset,     
    eval_dataset=eval_dataset,        
)

# Start training
trainer.train()

# Step 6: Evaluate using QWK (Quadratic Weighted Kappa)
def qwk_score(predictions, references):
    """
    Compute the Quadratic Weighted Kappa (QWK) score.
    """
    # Convert predictions and references to 1D arrays if necessary
    predictions = predictions.argmax(axis=-1) if len(predictions.shape) > 1 else predictions
    references = references.argmax(axis=-1) if len(references.shape) > 1 else references
    return cohen_kappa_score(predictions, references, weights='quadratic')

# Get model predictions on the evaluation dataset
predictions = trainer.predict(eval_dataset)

# If predictions.predictions is a tuple, extract the first element
predicted_logits = predictions.predictions[0] if isinstance(predictions.predictions, tuple) else predictions.predictions

# Get predicted labels by taking the index with the maximum logit for each sample
predicted_labels = predicted_logits.argmax(axis=-1)

# Get the true labels from the evaluation dataset
true_labels = eval_dataset["labels"]

# Compute QWK
qwk = qwk_score(predicted_labels, true_labels)
print(f"QWK: {qwk}")
