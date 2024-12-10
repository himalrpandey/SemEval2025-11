import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score, precision_score, f1_score, confusion_matrix, accuracy_score

import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModelForSequenceClassification

import os

# Constants
EMOTIONS = ['Anger', 'Fear', 'Joy', 'Sadness', 'Surprise']
#THRESHOLDS = [0.35, 0.45, 0.4, 0.55, 0.3] doing it dynamically right now
BATCH_SIZE = 16
EPOCHS = 20
NUM_WORKERS = 2  # Number of workers for DataLoader
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EARLY_STOPPING_PATIENCE = 5
BEST_THRESHOLDS = {}  # To store the optimized thresholds

# Set PyTorch to use multiple threads
torch.set_num_threads(NUM_WORKERS)

# Load Data
train = pd.read_csv('public_data/train/track_a/eng.csv')

# Split the data (80% training, 20% validation)
train_split, val_split = train_test_split(
    train,
    test_size=0.2,
    random_state=42,
    stratify=train[EMOTIONS].values.sum(axis=1)  # Maintain label distribution
)

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
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-multilingual-cased")
    model = AutoModelForSequenceClassification.from_pretrained(
        "distilbert-base-multilingual-cased", num_labels=1  # Binary classification
    )
    return tokenizer, model

# Train Function with Early Stopping
def train_model(model, train_loader, val_loader, device, epochs, patience=EARLY_STOPPING_PATIENCE, accumulation_steps=1):
    optimizer = AdamW(model.parameters(), lr=5e-5)
    criterion = nn.BCEWithLogitsLoss()
    model.to(device)

    best_loss = float('inf')
    no_improve_epochs = 0
    best_model = None

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        optimizer.zero_grad()

        for i, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].unsqueeze(1).to(device)

            # Forward pass
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(outputs.logits, labels)
            loss.backward()

            # Accumulate gradients
            if (i + 1) % accumulation_steps == 0 or (i + 1) == len(train_loader):
                optimizer.step()
                optimizer.zero_grad()

            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)

        # Validation loss for early stopping
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].unsqueeze(1).to(device)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                loss = criterion(outputs.logits, labels)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)

        print(f"Epoch {epoch + 1}, Training Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}")

        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            no_improve_epochs = 0
            best_model = model.state_dict()
        else:
            no_improve_epochs += 1
            if no_improve_epochs >= patience:
                print("Early stopping triggered.")
                model.load_state_dict(best_model)
                break

    return model

# Optimize Thresholds
def optimize_thresholds(y_true, y_probs):
    best_threshold = 0.5
    best_f1 = 0

    for threshold in np.arange(0.1, 0.9, 0.05):
        y_pred = (y_probs > threshold).astype(int)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold

    return best_threshold

# Get Predictions
def get_predictions(data_loader, model, device):
    model.eval()
    predictions = []
    true_labels = []

    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            probs = torch.sigmoid(logits).cpu().numpy()

            predictions.extend(probs.flatten())
            true_labels.extend(labels.cpu().numpy())

    return np.array(predictions), np.array(true_labels)

def print_confusion_matrix(y_true, y_pred, emotion):
    cm = confusion_matrix(y_true, y_pred)
    print(f"\nConfusion Matrix for {emotion}:")
    print(f"True Negatives: {cm[0, 0]}, False Positives: {cm[0, 1]}")
    print(f"False Negatives: {cm[1, 0]}, True Positives: {cm[1, 1]}")
    print(cm)
    
# Evaluation
def evaluate_predictions(y_true, y_pred, emotion, output_file):
    accuracy = accuracy_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred, zero_division=0)
    precision = precision_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    print_confusion_matrix(y_true, y_pred, emotion)

    with open(output_file, "a") as f:
        f.write(f"*** {emotion} ***\n")
        f.write(f"Accuracy: {accuracy:.4f}\n")
        f.write(f"Recall: {recall:.4f}, Precision: {precision:.4f}, F1: {f1:.4f}\n")

    print(f"*** {emotion} ***")
    print(f"Accuracy: {accuracy:.4f}, Recall: {recall:.4f}, Precision: {precision:.4f}, F1: {f1:.4f}")

# Main function
def main():
    predictions = {}
    output_file = "results_summary.txt"

    # Clear the results file
    if os.path.exists(output_file):
        os.remove(output_file)

    # Prepare final predictions dataframe
    final_predictions = pd.DataFrame()

    for emotion in EMOTIONS:
        print(f"\nProcessing emotion: {emotion}")

        # Prepare data for the emotion
        train_texts, train_labels = train_split["text"].tolist(), train_split[emotion].values
        val_texts, val_labels = val_split["text"].tolist(), val_split[emotion].values
        val_ids = val_split["id"].tolist()  # Ensure 'id' column is included in the data

        # Initialize tokenizer and model
        tokenizer, model = initialize_model_and_tokenizer()

        # Create datasets
        train_dataset = EmotionDataset(train_texts, train_labels, tokenizer)
        val_dataset = EmotionDataset(val_texts, val_labels, tokenizer)

        # Create DataLoaders
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

        # Train the model
        print(f"{emotion}...")
        model = train_model(model, train_loader, val_loader, DEVICE, EPOCHS)

        # Generate predictions
        y_probs, y_true = get_predictions(val_loader, model, DEVICE)
        best_threshold = optimize_thresholds(y_true, y_probs)
        BEST_THRESHOLDS[emotion] = best_threshold

        # Predict using optimized threshold
        y_pred = (y_probs > best_threshold).astype(int)

        # Store predictions for this emotion
        predictions[emotion] = y_pred

        # Evaluate
        evaluate_predictions(y_true, y_pred, emotion, output_file)

        # Add predictions for this emotion to the final dataframe
        if final_predictions.empty:
            final_predictions["id"] = val_ids  # Add 'id' column once
        final_predictions[emotion] = y_pred

    # Save predictions in the Codabench format
    output_csv_file = "pred_eng_a.csv"  # For track A English
    final_predictions.to_csv(output_csv_file, index=False)
    print(f"\nPredictions saved to '{output_csv_file}'.")

    # Print best thresholds
    print("\nBest Thresholds:")
    for emotion, threshold in BEST_THRESHOLDS.items():
        print(f"{emotion}: {threshold:.2f}")

if __name__ == "__main__":
    main()
