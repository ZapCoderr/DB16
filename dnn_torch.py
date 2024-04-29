import argparse
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score

# Setup argparse for command line arguments
parser = argparse.ArgumentParser(description="Training and evaluation of a sequence-based neural network with attention")
parser.add_argument("--train_input_file", type=str, default="train_sample.csv", help="File path for training data")
parser.add_argument("--test_input_file", type=str, default="test_sample.csv", help="File path for testing data")
parser.add_argument("--checkpointDir", default="ckpt_dir")
parser.add_argument("--init_checkpoint", default=None)
parser.add_argument("--max_seq_length", default=10, type=int)
parser.add_argument("--do_train", action='store_true')
parser.add_argument("--do_eval", action='store_true')
parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs")
parser.add_argument("--batch_size", type=int, default=256, help="Batch size for training")
parser.add_argument("--learning_rate", type=float, default=0.0005, help="Learning rate for optimizer")
parser.add_argument("--num_train_steps", default=1000000, type=int)
parser.add_argument("--num_warmup_steps", default=100, type=int)
parser.add_argument("--save_checkpoints_steps", default=8000, type=int)
parser.add_argument("--dropout_rate", default=0.2, type=float)
parser.add_argument("--model", default="dnn")
args = parser.parse_args()

# Define the Dataset
class CustomDataset(Dataset):
    def __init__(self, filepath, epoch=1, batch_size=256):
        self.data = pd.read_csv(filepath, header=None)
        self.n_channel = 200
        self.d_channel = 8
        self.n_coin = 378
        self.d_coin = 8
        self.n_seq_feat = 147
        self.n_target_feat = 138
        self.max_length = 50
        self.feat_config = {
            "n_channel": 200,
            "d_channel": 8,
            "n_coin": 378,
            "d_coin": 8,
            "n_seq_feat": 147,
            "n_target_feat": 138,
            "max_length": 50,
            "batch_size": batch_size,
            "epoch": epoch,
        }
        # Pre-process sequence data
        self.label, self.channel, self.channel_id, self.coin, self.coin_id, self.time_stamp, self.length, self.coin_id_seq, self.feature_seq, self.feature_target = self.parse_split(self.data)
        self.seq_embedding, self.labels = self.process_data()
        self.seq_coin_embedding = self.parse_sequence()

    def parse_split(self, df):
        label = torch.tensor(df.iloc[:, 0].values)
        channel = torch.tensor(df.iloc[:, 1].values)
        channel_id = torch.tensor(df.iloc[:, 2].values)
        coin = df.iloc[:, 3].values
        coin_id = torch.tensor(df.iloc[:, 4].values)
        time_stamp = torch.tensor(df.iloc[:, 5].values)
        length = torch.tensor(df.iloc[:, 6].values, dtype=torch.float32)
        coin_id_seq = df.iloc[:, 7].values
        feature_seq = df.iloc[:, 8].values
        feature_target = torch.tensor(df.iloc[:, 9:].values)

        return label, channel, channel_id, coin, coin_id, time_stamp, length, coin_id_seq, feature_seq, feature_target

    def process_string(self, s):
        segments = s.split('\t')
        flat_list = [float(num) for segment in segments for num in segment.split('')]
        adjusted_list = flat_list[:147] if len(flat_list) > 147 else flat_list + [0.0] * (147 - len(flat_list))
        return adjusted_list

    def process_data(self):
        labels = torch.tensor(self.data.iloc[:, 0].values, dtype=torch.float32)
        seq_embedding = torch.tensor(self.data.iloc[:, 8].apply(self.process_string).tolist())
        return seq_embedding, labels

    def parse_sequence(self):
        sequences = self.data.iloc[:, 7].apply(lambda x: [float(num) for num in x.split('\t')]).tolist()

        processed_sequences = []
        for seq in sequences:
            tensor_seq = torch.tensor(seq)
            if len(seq) < 10:
                padded_seq = F.pad(tensor_seq, (0, 10 - len(seq)), "constant", 0)
            else:
                padded_seq = tensor_seq[:10]
            processed_sequences.append(padded_seq)

        fixed_length_sequences = torch.stack(processed_sequences)

        return fixed_length_sequences

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.length[idx], self.seq_embedding[idx], self.seq_coin_embedding[idx], self.labels[idx]

# DNN
class DNN(nn.Module):
    def __init__(self, input_dim):
        super(DNN, self).__init__()
        self.input_dim = input_dim
        self.embedding_dim = 1 
        self.fc1 = nn.Linear(input_dim, self.embedding_dim)
        self.fc2 = nn.Linear(self.embedding_dim, 32)
        self.output = nn.Linear(32, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.sigmoid(self.output(x))
        return x

# Training function
def train(model, loader, optimizer, criterion, epochs):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for length, seq_embedding, seq_coin_embedding, targets in loader:
            length = length.unsqueeze(1)
            optimizer.zero_grad()
            outputs = model(torch.cat((length, seq_embedding, seq_coin_embedding), dim=1))
            loss = criterion(outputs.squeeze(), targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss / len(loader)}")

# Evaluation function
def accuracy(model, loader):
    model.eval()
    predictions, actuals = [], []
    with torch.no_grad():
        for length, seq_embedding, seq_coin_embedding, targets in loader:
            length = length.unsqueeze(1)
            outputs = model(torch.cat((length, seq_embedding, seq_coin_embedding), dim=1))
            predicted = outputs.round().squeeze()
            predictions.extend(predicted.numpy())
            actuals.extend(targets.numpy())
    accuracy = np.mean(np.array(predictions) == np.array(actuals))
    return accuracy

def RatioHit(predictions, actuals, k):
    indexed_predictions = list(enumerate(predictions))
    indexed_predictions.sort(key=lambda x: x[1], reverse=True)
    top_k_predictions = indexed_predictions[:k]
    hits = sum(actuals[idx] for idx, _ in top_k_predictions)
    hit_rate = hits / k
    return hit_rate


def evaluate(model, loader, k_values=[1, 3, 5, 10, 20, 30]):
    model.eval()
    total_hits = {k: 0 for k in k_values}
    total_counts = {k: 0 for k in k_values}
    
    with torch.no_grad():
        for length, seq_embedding, seq_coin_embedding, targets in loader:
            length = length.unsqueeze(1)
            outputs = model(torch.cat((length, seq_embedding, seq_coin_embedding), dim=1))
            probabilities = torch.sigmoid(outputs).numpy()
            actuals = targets.numpy()

            for k in k_values:
                if len(probabilities) >= k: 
                    hit_rate = RatioHit(probabilities, actuals, k)
                    total_hits[k] += hit_rate
                    total_counts[k] += 1

    average_hits = {k: total_hits[k] * 10 / total_counts[k] * k for k in k_values if total_counts[k] > 0}
    return average_hits


def main():
    train_dataset = CustomDataset(args.train_input_file, args.epochs, args.batch_size)
    test_dataset = CustomDataset(args.test_input_file, args.epochs, args.batch_size)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    model = DNN(input_dim=1+train_dataset.seq_embedding.shape[1]+train_dataset.seq_coin_embedding.shape[1])
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = nn.BCELoss()

    train(model, train_loader, optimizer, criterion, args.epochs)
    acc = accuracy(model, test_loader)
    print(f"Accuracy: {acc:.4f}")

    if True:
        hit_rates = evaluate(model, test_loader)
        for k, rate in hit_rates.items():
            print(f"Hit Rate at top {k}: {rate:.4f}")

if __name__ == "__main__":
    main()
