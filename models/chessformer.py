import torch.nn as nn
from transformers import T5ForConditionalGeneration

class Chessformer(nn.Module):
    def __init__(self, move_vocab_size):
        super().__init__()
        self.model = T5ForConditionalGeneration.from_pretrained("t5-small")
        self.move_head = nn.Linear(self.model.config.d_model, move_vocab_size)

    def forward(self, input_ids, attention_mask, decoder_input_ids, explanation_labels, move_label):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            labels=explanation_labels,
        )

        move_logits = self.move_head(outputs.encoder_last_hidden_state[:, 0, :])
        return move_logits, outputs.loss
