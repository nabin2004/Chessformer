from torch.utils.data import Dataset
import json

class ChessDataset(Dataset):
    def __init__(self, path, tokenizer, move2id):
        with open(path) as f:
            self.samples = [json.loads(line) for line in f]
        self.tokenizer = tokenizer
        self.move2id = move2id

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        inputs = self.tokenizer(sample["fen"], truncation=True, padding="max_length", return_tensors="pt")
        labels = self.tokenizer(sample["explanation"], truncation=True, padding="max_length", return_tensors="pt")
        move_label = self.move2id[sample["move"]]
        return {
            "input_ids": inputs["input_ids"].squeeze(),
            "attention_mask": inputs["attention_mask"].squeeze(),
            "decoder_input_ids": labels["input_ids"].squeeze(),
            "explanation_labels": labels["input_ids"].squeeze(),
            "move_label": move_label
        }
