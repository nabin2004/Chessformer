def predict(model, tokenizer, fen, move_vocab):
    inputs = tokenizer(fen, return_tensors="pt")
    generated = model.model.generate(**inputs)
    explanation = tokenizer.decode(generated[0], skip_special_tokens=True)

    with torch.no_grad():
        output = model.model.encoder(**inputs)
        move_logits = model.move_head(output.last_hidden_state[:, 0, :])
        move_idx = move_logits.argmax(dim=1).item()
        move = move_vocab[move_idx]

    return move, explanation
