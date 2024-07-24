import torch
import torch.nn as nn
import torch.utils.data as data
import torch.optim as optim

import ast
import os
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import re
import sys
import pytorch_lightning as pl

class RnnLipModel(nn.Module):
    def __init__(self, input_dim=40, hidden_dim=40, output_dim=5, num_layers=5):
        super().__init__()
        self.num_layers = num_layers
        self.output_dim = output_dim

        self.encoder = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, bidirectional=False, dropout=0.6)
        self.decoder = nn.LSTM(output_dim, hidden_dim, num_layers, batch_first=True, bidirectional=False, dropout=0.6)
        self.fc = nn.Linear(hidden_dim, output_dim)

        self.encoder_dropout = nn.Dropout(0.2)
        self.decoder_dropout = nn.Dropout(0.2)

    def forward(self, x):
        _, (h, c) = self.encoder(x)
        h = self.encoder_dropout(h)
        c = self.encoder_dropout(c)

        decoder_input = torch.zeros(x.size(0), x.size(1), self.output_dim, device=x.device)
        output, _ = self.decoder(decoder_input, (h, c))
        output = self.fc(output)
        return output

# ---
EPOCHS = 10000
BATCH_SIZE = 50
LEARNING_RATE = 0.0002
SHUFFLE_TRAIN = True

DATA_FOLDER = '../data'
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print(f'Device: {device}\n')

# ---

model = RnnLipModel().to(device)


# GET DATA
def get_arr(txt): # with data folder
    txt = os.path.join(DATA_FOLDER, txt)
    arr = open(txt, 'r').read()
    arr = ast.literal_eval(arr)
    arr = np.array(arr)
    return arr

def get_file(txt): # without data folder
    arr = open(txt, 'r').read()
    arr = ast.literal_eval(arr)
    arr = np.array(arr)
    return arr

def numeric_sort_key(s):
    match = re.search(r'^(\d+)', s)
    return int(match.group(1)) if match else float('inf')

input_arr = sorted([f for f in os.listdir(DATA_FOLDER) if f.endswith('inputs.txt')], key=numeric_sort_key)
label_arr = sorted([f for f in os.listdir(DATA_FOLDER) if f.endswith('labels.txt')], key=numeric_sort_key)

def get_infer_tuple(infer_folder):
    infer_input_arr = sorted([f for f in os.listdir(infer_folder) if f.endswith('inputs.txt')], key=numeric_sort_key)
    infer_label_arr = sorted([f for f in os.listdir(infer_folder) if f.endswith('labels.txt')], key=numeric_sort_key)
    infer_tuple = tuple(zip(infer_input_arr, infer_label_arr))
    return infer_tuple

def get_dataloader(model_inputs, model_labels):
    train_inputs, test_inputs, train_labels, test_labels = train_test_split(model_inputs, model_labels, test_size=0.1)
    train_inputs = torch.tensor(train_inputs, dtype=torch.float32)
    train_labels = torch.tensor(train_labels, dtype=torch.float32)
    test_inputs = torch.tensor(test_inputs, dtype=torch.float32)
    test_labels = torch.tensor(test_labels, dtype=torch.float32)

    train_dataset = data.TensorDataset(train_inputs, train_labels)
    test_dataset = data.TensorDataset(test_inputs, test_labels)
    train_loader = data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=SHUFFLE_TRAIN)
    test_loader = data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    return train_loader, test_loader

# ---
# TRAINING
train_loss_history = []
val_loss_history = []
train_accuracy_history = []
val_accuracy_history = []

def train(model, optimizer, train_loader, test_loader):
    criterion = nn.MSELoss()

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        correct_train = 0
        total_train = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * inputs.size(0)
            correct_train += (outputs.round() == labels).sum().item()
            total_train += labels.size(0)

        train_loss = train_loss / len(train_loader.dataset)
        train_accuracy = correct_train / total_train
        train_loss_history.append(train_loss)
        train_accuracy_history.append(train_accuracy)

        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                correct_val += (outputs.round() == labels).sum().item()
                total_val += labels.size(0)

        val_loss = val_loss / len(test_loader.dataset)
        val_accuracy = correct_val / total_val
        val_loss_history.append(val_loss)
        val_accuracy_history.append(val_accuracy)

        print(f'Epoch {epoch+1}/{EPOCHS}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}')

    plt.figure(figsize=(10, 6))
    plt.plot(train_loss_history, label='Train Loss')
    plt.plot(val_loss_history, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(f'loss_{EPOCHS}_{BATCH_SIZE}.png')
    plt.show()

def main_train():
    all_inputs = []
    all_labels = []

    for input in input_arr:
        input = get_arr(input)
        all_inputs.append(input)

    for label in label_arr:
        label = get_arr(label)
        all_labels.append(label)
    
    all_inputs = np.array(all_inputs)
    all_labels = np.array(all_labels)

    train_loader, test_loader = get_dataloader(all_inputs, all_labels)
    model = RnnLipModel().to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    train(model, optimizer, train_loader, test_loader)

    weights_path = 'lip_model_seq2seq.pth'
    torch.save(model.state_dict(), weights_path)
    print(f'Saved model weights to {weights_path}')
    torch.save(model, 'rnn_model.pth')

if __name__ == '__main__':
    main_train()
