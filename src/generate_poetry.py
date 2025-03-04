"""This module ships a function."""

import torch

from util import convert


def sample_stored_model(model_path):
    """Sample from a stored model."""
    with open(model_path, "rb") as weight_file:
        model, args, data_loader = torch.load(
            weight_file, map_location=torch.device("cpu"), weights_only=False
        )

    model.eval()
    print("args", args)
    seq_len = 256
    init_list = [0, 1, 3, 9, 13, 24, 29, 42]
    # init_list = [0, 1, 13]
    init_chars = torch.tensor(init_list)
    init_chars = (
        torch.nn.functional.one_hot(init_chars, num_classes=data_loader.vocab_size)
        .unsqueeze(1)
        .type(torch.float32)
    )
    print(init_chars.shape)

    sequences = model.sample(init_chars, seq_len)
    sequences = torch.argmax(sequences, -1)

    print(data_loader.inv_vocab)

    seq_conv = convert(sequences, data_loader.inv_vocab)
    for i in range(len(init_list)):
        input = data_loader.inv_vocab[init_list[i]]
        print('--> Input "{}" leads to poetry:'.format(input))
        print(input + "".join(seq_conv[i]))
        print("")


if __name__ == "__main__":
    model_path = "./saved/net_save.pkl"
    sample_stored_model(model_path)
