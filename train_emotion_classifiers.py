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
EMOTIONS = ['Joy', 'Sadness', 'Surprise', 'Fear', 'Anger']
THRESHOLDS = [0.35, 0.45, 0.4, 0.55, 0.3]
BATCH_SIZE = 8
EPOCHS = 1
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load Data
print("Loading data...")
train = pd.read_csv('public_data/train/track_a/eng.csv')

# Reduce the dataset size for faster testing
train = train.sample(n=500, random_state=42).reset_index(drop=True)

# Split the data (80% training, 20% validation)
print("Splitting data...")
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

# Train Function
def train_model(model, train_loader, val_loader, device, epochs):
    optimizer = AdamW(model.parameters(), lr=5e-5)
    criterion = nn.BCEWithLogitsLoss()
    model.to(device)

    for epoch in range(epochs):
        model.train()
        train_loss = 0

        for batch in train_loader:
            optimizer.zero_grad()

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].unsqueeze(1).to(device)  # Adjust shape for binary output

            # Forward pass
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(outputs.logits, labels)

            # Backward pass
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        print(f"Epoch {epoch + 1}, Training Loss: {avg_train_loss:.4f}")

# Get Predictions
def get_predictions(data_loader, model, device, threshold):
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
            preds = (torch.sigmoid(logits).cpu().numpy() > threshold).astype(int)

            predictions.extend(preds.flatten())  # Flatten to 1D array
            true_labels.extend(labels.cpu().numpy())

    return np.array(predictions), np.array(true_labels)

# Evaluation
def evaluate_predictions(y_true, y_pred, emotion, output_file):
    accuracy = accuracy_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred, zero_division=0)
    precision = precision_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    with open(output_file, "a") as f:
        f.write(f"*** {emotion} ***\n")
        f.write(f"Accuracy: {accuracy:.4f}\n")
        f.write(f"Recall: {recall:.4f}, Precision: {precision:.4f}, F1: {f1:.4f}\n")

    print(f"*** {emotion} ***")
    print(f"Accuracy: {accuracy:.4f}, Recall: {recall:.4f}, Precision: {precision:.4f}, F1: {f1:.4f}")

# Confusion Matrix
def plot_confusion_matrix(y_true, y_pred, emotion, output_file):
    cm = confusion_matrix(y_true, y_pred)
    with open(output_file, "a") as f:
        f.write(f"\nConfusion Matrix for {emotion}:\n{cm}\n")
        f.write(f"True Negatives: {cm[0, 0]}, False Positives: {cm[0, 1]}\n")
        f.write(f"False Negatives: {cm[1, 0]}, True Positives: {cm[1, 1]}\n\n")

# Main function
def main():
    models = {}
    predictions = {}
    output_file = "results_summary.txt"

    # Clear the results file
    if os.path.exists(output_file):
        os.remove(output_file)

    for emotion, threshold in zip(EMOTIONS, THRESHOLDS):
        print(f"\nProcessing emotion: {emotion}")

        # Prepare data for the emotion
        train_texts, train_labels = train_split['text'].tolist(), train_split[emotion].values
        val_texts, val_labels = val_split['text'].tolist(), val_split[emotion].values

        # Initialize tokenizer and model
        tokenizer, model = initialize_model_and_tokenizer()

        # Create datasets
        train_dataset = EmotionDataset(train_texts, train_labels, tokenizer)
        val_dataset = EmotionDataset(val_texts, val_labels, tokenizer)

        # Create DataLoaders
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

        # Train the model
        print(f"Training model for {emotion}...")
        train_model(model, train_loader, val_loader, DEVICE, EPOCHS)
        models[emotion] = model

        # Generate predictions
        print(f"Generating predictions for {emotion}...")
        val_dataset = EmotionDataset(val_texts, val_labels, tokenizer)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
        y_pred, y_true = get_predictions(val_loader, model, DEVICE, THRESHOLDS[EMOTIONS.index(emotion)])

        # Store predictions for this emotion
        predictions[emotion] = y_pred

        # Evaluate
        evaluate_predictions(y_true, y_pred, emotion, output_file)
        plot_confusion_matrix(y_true, y_pred, emotion, output_file)

    # Save predictions to CSV
    print("\nSaving predictions...")
    val_predictions = val_split.copy()
    for emotion in EMOTIONS:
        val_predictions[emotion] = predictions[emotion]

    val_predictions.to_csv('val_predictions_with_text.csv', index=False)
    print("\nValidation predictions saved to 'val_predictions_with_text.csv'.")

if __name__ == "__main__":
    main()
