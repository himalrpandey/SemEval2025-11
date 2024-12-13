import pandas as pd
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix

EMOTIONS = ['Anger', 'Fear', 'Joy', 'Sadness', 'Surprise']
#Threshold proportional to positive labeling in dataset of 2768 inputs. {Joy: 674, Anger: 333, Sadness: 878, Surprise: 839, Fear: 1611}
THRESHOLDS = {'Joy': 0.24, 'Anger': 0.12, 'Sadness': 0.32, 'Surprise': 0.30, 'Fear': 0.58}
BATCH_SIZE = 32
EPOCHS = 15
NUM_WORKERS = 1  # Number of workers for DataLoader and gpus
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EARLY_STOPPING_PATIENCE = 5

torch.set_num_threads(NUM_WORKERS)

train = pd.read_csv('public_data/train/track_a/eng.csv')
dev = pd.read_csv('public_data/train/track_a/rus.csv')

class EmotionDataset(Dataset):
    """
    This class is used to create a PyTorch dataset from the texts and labels.
    """
    def __init__(self, texts, labels, tokenizer, max_length=256):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    """
    This function returns the input_ids, attention_mask, and labels for the given index.
    """
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
        
"""
This function initializes the model and tokenizer. 
"""
def initialize_model_and_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-multilingual-cased")
    model = AutoModelForSequenceClassification.from_pretrained(
        "distilbert-base-multilingual-cased", num_labels=1
    )
    return tokenizer, model

"""
This function trains the model.
args:
    model: The model to train
    train_loader: DataLoader for training data
    device: Device to use for training
    epochs: Number of epochs to train
    patience: Number of epochs to wait for improvement before early stopping
returns:
    model: Trained model
"""
def train_model(model, train_loader, device, epochs, patience=EARLY_STOPPING_PATIENCE):
    optimizer = AdamW(model.parameters(), lr=1e-5)
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
                print("Early stopped")
                model.load_state_dict(best_model)
                break

    return model

"""
This function gets predictions from the model.
args:
    data_loader: DataLoader for the data
    model: Model to get predictions from
    device: Device to use for predictions
returns:
    predictions: Predictions from the model
"""
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

"""
This function prints the confusion matrix for the given emotion.
"""
def print_confusion_matrix(y_true, y_pred, emotion):
    cm = confusion_matrix(y_true, y_pred)
    print(f"\nConfusion Matrix for {emotion}:")
    print(f"True Negatives: {cm[0, 0]}, False Positives: {cm[0, 1]}")
    print(f"False Negatives: {cm[1, 0]}, True Positives: {cm[1, 1]}")
    print(cm)

"""
This function evaluates the predictions for the given emotion.
It compares the true labels with the predicted labels and prints the accuracy, recall, precision, and F1 score.
"""
def evaluate_predictions(y_true, y_pred, emotion):
    accuracy = accuracy_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred, zero_division=0)
    precision = precision_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    print_confusion_matrix(y_true, y_pred, emotion)

    print(f"{emotion}")
    print(f"Accuracy: {accuracy:.4f}, Recall: {recall:.4f}, Precision: {precision:.4f}, F1: {f1:.4f}")



def main():
    final_predictions = pd.DataFrame()
    final_predictions["id"] = dev["id"]

    for emotion in EMOTIONS:
        print(f"\n{emotion}")

        train_texts, train_labels = train["text"].tolist(), train[emotion].values
        dev_texts = dev["text"].tolist()
        dev_labels = dev[emotion].values

        tokenizer, model = initialize_model_and_tokenizer()

        train_dataset = EmotionDataset(train_texts, train_labels, tokenizer)
        dev_dataset = EmotionDataset(dev_texts, np.zeros(len(dev_texts)), tokenizer)

        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
        dev_loader = DataLoader(dev_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

        model = train_model(model, train_loader, DEVICE, EPOCHS)

        y_probs = get_predictions(dev_loader, model, DEVICE)

        threshold = THRESHOLDS[emotion]
        y_pred = (y_probs > threshold).astype(int)
        final_predictions[emotion] = y_pred
        
        evaluate_predictions(dev_labels, y_pred, emotion)

    # Save predictions the format for the comp
    #currently checking predictions for russian train dataset
    output_csv_file = "pred_rus.csv"
    final_predictions.to_csv(output_csv_file, index=False)
    print(f"\nPredictions saved to {output_csv_file}")

if __name__ == "__main__":
    main()
