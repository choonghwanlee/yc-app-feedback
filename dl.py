from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import cohen_kappa_score
import numpy as np
import torch.optim.lr_scheduler as lr_scheduler
from datasets import load_dataset

# 1. Dataset Preparation
class PitchDataset(Dataset):
    def __init__(self, texts, clarity_scores, team_scores, traction_scores, tokenizer, max_length=512):
        self.texts = texts
        self.clarity_scores = clarity_scores
        self.team_scores = team_scores
        self.traction_scores = traction_scores
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.texts)
        
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        # Convert scores to zero-based index (1-5 → 0-4)
        clarity_score = self.clarity_scores[idx] - 1
        team_score = self.team_scores[idx] - 1
        traction_score = self.traction_scores[idx] - 1
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'clarity_score': torch.tensor(clarity_score, dtype=torch.long),
            'team_score': torch.tensor(team_score, dtype=torch.long),
            'traction_score': torch.tensor(traction_score, dtype=torch.long)
        }

# 2. Model Definition (Using Bert model with gradient checkpointing)
class PitchEvaluationModel(nn.Module):
    def __init__(self, pretrained_model="bert-base-uncased", use_gradient_checkpointing=True):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(pretrained_model)
        if use_gradient_checkpointing:
            self.encoder.gradient_checkpointing_enable()
        self.dropout = nn.Dropout(0.3)
        self.clarity_classifier = nn.Linear(self.encoder.config.hidden_size, 5)
        self.team_classifier = nn.Linear(self.encoder.config.hidden_size, 5)
        self.traction_classifier = nn.Linear(self.encoder.config.hidden_size, 5)
        
    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0]  # Using CLS token
        pooled_output = self.dropout(pooled_output)
        
        clarity_logits = self.clarity_classifier(pooled_output)
        team_logits = self.team_classifier(pooled_output)
        traction_logits = self.traction_classifier(pooled_output)
        
        return clarity_logits, team_logits, traction_logits

# 3. Training Function (Includes learning rate scheduling and multi-task weighted loss)
def train_model(model, train_loader, val_loader, device, epochs=10,
                clarity_weight=1.0, team_weight=1.0, traction_weight=1.0):
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    # Use StepLR, decrease learning rate every 2 epochs
    scheduler = lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.8)
    criterion = nn.CrossEntropyLoss()
    best_qwk = 0

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        for batch in train_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            clarity_scores = batch['clarity_score'].to(device)
            team_scores = batch['team_score'].to(device)
            traction_scores = batch['traction_score'].to(device)
            
            optimizer.zero_grad()
            
            clarity_logits, team_logits, traction_logits = model(input_ids, attention_mask)
            
            clarity_loss = criterion(clarity_logits, clarity_scores)
            team_loss = criterion(team_logits, team_scores)
            traction_loss = criterion(traction_logits, traction_scores)
            
            # Multi-task weighted loss
            loss = clarity_weight * clarity_loss + team_weight * team_loss + traction_weight * traction_loss
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        scheduler.step()  # Adjust learning rate
        
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
        
        # Evaluate on validation set and save the best model
        qwk = evaluate_model(model, val_loader, device)
        if qwk > best_qwk:
            best_qwk = qwk
            torch.save(model.state_dict(), "best_pitch_model.pt")
            print(f"Model saved with QWK: {best_qwk:.4f}")
    
    return model

# 4. Evaluation Function (Calculates QWK)
def evaluate_model(model, data_loader, device):
    model.eval()
    
    all_clarity_preds = []
    all_team_preds = []
    all_traction_preds = []
    
    all_clarity_true = []
    all_team_true = []
    all_traction_true = []
    
    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            clarity_logits, team_logits, traction_logits = model(input_ids, attention_mask)
            
            # Convert predictions back to 1-5 scale
            clarity_preds = torch.argmax(clarity_logits, dim=1).cpu().numpy() + 1
            team_preds = torch.argmax(team_logits, dim=1).cpu().numpy() + 1
            traction_preds = torch.argmax(traction_logits, dim=1).cpu().numpy() + 1
            
            all_clarity_preds.extend(clarity_preds)
            all_team_preds.extend(team_preds)
            all_traction_preds.extend(traction_preds)
            
            all_clarity_true.extend((batch['clarity_score'].cpu().numpy() + 1))
            all_team_true.extend((batch['team_score'].cpu().numpy() + 1))
            all_traction_true.extend((batch['traction_score'].cpu().numpy() + 1))
    
    clarity_qwk = cohen_kappa_score(all_clarity_true, all_clarity_preds, weights='quadratic')
    team_qwk = cohen_kappa_score(all_team_true, all_team_preds, weights='quadratic')
    traction_qwk = cohen_kappa_score(all_traction_true, all_traction_preds, weights='quadratic')
    
    overall_qwk = (clarity_qwk + team_qwk + traction_qwk) / 3
    
    print("Evaluation Results:")
    print(f"Clarity QWK: {clarity_qwk:.4f}")
    print(f"Team Market Fit QWK: {team_qwk:.4f}")
    print(f"Traction QWK: {traction_qwk:.4f}")
    print(f"Overall QWK: {overall_qwk:.4f}")
    
    return overall_qwk

# 5. Main Function (Includes training, validation, and test set evaluation)
def main():
    # Load data
    dataset = load_dataset("jasonhwan/yc-startup-pitches-with-scores", split="train")
    df = dataset.to_pandas()

    df.to_csv("yc_startup_pitches.csv", index=False)

    df = pd.read_csv("yc_startup_pitches.csv")
    
    # Extract text and scores
    texts = df['transcript'].values
    clarity_scores = df['clarity_score'].values
    team_scores = df['team_market_fit_score'].values
    traction_scores = df['traction_validation_score'].values
    
    # Split dataset: 70% training, 15% validation, 15% testing
    train_texts, temp_texts, train_clarity, temp_clarity, train_team, temp_team, train_traction, temp_traction = train_test_split(
        texts, clarity_scores, team_scores, traction_scores, test_size=0.3, random_state=42
    )
    val_texts, test_texts, val_clarity, test_clarity, val_team, test_team, val_traction, test_traction = train_test_split(
        temp_texts, temp_clarity, temp_team, temp_traction, test_size=0.5, random_state=42
    )
    
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained("bert-large-uncased")
    
    train_dataset = PitchDataset(train_texts, train_clarity, train_team, train_traction, tokenizer)
    val_dataset = PitchDataset(val_texts, val_clarity, val_team, val_traction, tokenizer)
    test_dataset = PitchDataset(test_texts, test_clarity, test_team, test_traction, tokenizer)
    
    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8)
    test_loader = DataLoader(test_dataset, batch_size=8)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = PitchEvaluationModel(pretrained_model="bert-base-uncased", use_gradient_checkpointing=True)
    model.to(device)
    
    model = train_model(model, train_loader, val_loader, device, epochs=10)
    
    model.load_state_dict(torch.load("best_pitch_model.pt"))

if __name__ == "__main__":
    main()
