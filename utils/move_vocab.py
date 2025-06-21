def build_move_vocab(data_path):
    moves = set()
    with open(data_path) as f:
        for line in f:
            sample = json.loads(line)
            moves.add(sample["move"])
    move2id = {move: i for i, move in enumerate(sorted(moves))}
    id2move = {i: move for move, i in move2id.items()}
    return move2id, id2move
