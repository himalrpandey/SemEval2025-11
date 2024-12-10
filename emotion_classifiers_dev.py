import pandas as pd
import numpy as np
from sklearn.metrics import recall_score, precision_score, f1_score, confusion_matrix, accuracy_score

import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModelForSequenceClassification

import os

# Constants
EMOTIONS = ['Anger', 'Fear', 'Joy', 'Sadness', 'Surprise']
THRESHOLDS = {'Joy': 0.10, 'Anger': 0.15, 'Sadness': 0.85, 'Surprise': 0.30, 'Fear': 0.25}
BATCH_SIZE = 16
EPOCHS = 20
NUM_WORKERS = 4  # Number of workers for DataLoader
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EARLY_STOPPING_PATIENCE = 5
BEST_THRESHOLDS = {}  # To store the optimized thresholds

# Set PyTorch to use multiple threads
torch.set_num_threads(NUM_WORKERS)

# Load Data
train = pd.read_csv('public_data/train/track_a/eng.csv')
dev = pd.read_csv('public_data/dev/track_a/eng_a.csv')

# Dataset class
class EmotionDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(label, dtype=torch.float),
        }

# Initialize tokenizer and model
def initialize_model_and_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-cased")
    model = AutoModelForSequenceClassification.from_pretrained(
        "distilbert-base-cased", num_labels=1  # Binary classification
    )
    return tokenizer, model

# Train Function with Early Stopping
def train_model(model, train_loader, device, epochs, patience=EARLY_STOPPING_PATIENCE):
    optimizer = AdamW(model.parameters(), lr=5e-5)
    criterion = nn.BCEWithLogitsLoss()
    model.to(device)

    best_loss = float('inf')
    no_improve_epochs = 0
    best_model = None

    for epoch in range(epochs):
        model.train()
        train_loss = 0

        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].unsqueeze(1).to(device)

            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(outputs.logits, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        print(f"Epoch {epoch + 1}, Training Loss: {avg_train_loss:.4f}")

        if avg_train_loss < best_loss:
            best_loss = avg_train_loss
            no_improve_epochs = 0
            best_model = model.state_dict()
        else:
            no_improve_epochs += 1
            if no_improve_epochs >= patience:
                print("Early stopping triggered.")
                model.load_state_dict(best_model)
                break

    return model

# Get Predictions
def get_predictions(data_loader, model, device):
    model.eval()
    predictions = []

    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            probs = torch.sigmoid(logits).cpu().numpy()

            predictions.extend(probs.flatten())

    return np.array(predictions)

# Main function
def main():
    # Prepare final predictions dataframe
    final_predictions = pd.DataFrame()
    final_predictions["id"] = dev["id"]  # Include 'id' column from the dev dataset

    for emotion in EMOTIONS:
        print(f"\nProcessing emotion: {emotion}")

        # Prepare training data
        train_texts, train_labels = train["text"].tolist(), train[emotion].values
        dev_texts = dev["text"].tolist()

        # Initialize tokenizer and model
        tokenizer, model = initialize_model_and_tokenizer()

        # Create datasets
        train_dataset = EmotionDataset(train_texts, train_labels, tokenizer)
        dev_dataset = EmotionDataset(dev_texts, np.zeros(len(dev_texts)), tokenizer)  # Placeholder labels for dev

        # Create DataLoaders
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
        dev_loader = DataLoader(dev_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

        # Train the model
        print(f"Training model for {emotion}...")
        model = train_model(model, train_loader, DEVICE, EPOCHS)

        # Generate predictions
        y_probs = get_predictions(dev_loader, model, DEVICE)

        # Use a threshold provided from training for each emotion for predictions
        threshold = THRESHOLDS[emotion]
        y_pred = (y_probs > threshold).astype(int)
        final_predictions[emotion] = y_pred

    # Save predictions in the Codabench format
    output_csv_file = "pred_eng_a.csv"  # Filename for Codabench
    final_predictions.to_csv(output_csv_file, index=False)
    print(f"\nPredictions saved to '{output_csv_file}'.")

if __name__ == "__main__":
    main()
