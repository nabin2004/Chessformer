from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader

def train(model, dataloader, epochs=3):
    optimizer = Adam(model.parameters(), lr=5e-5)
    move_loss_fn = CrossEntropyLoss()

    for epoch in range(epochs):
        for batch in dataloader:
            move_logits, explanation_loss = model(
                batch["input_ids"],
                batch["attention_mask"],
                batch["decoder_input_ids"],
                batch["explanation_labels"],
                batch["move_label"]
            )
            move_loss = move_loss_fn(move_logits, batch["move_label"])
            total_loss = move_loss + explanation_loss

            total_loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            print(f"Loss: {total_loss.item():.4f}")
