"""Talk to your model."""

import torch

from util import convert

if __name__ == "__main__":
    model_path = "./saved/net_save.pkl"
    with open(model_path, "rb") as weight_file:
        model, args, data_loader = torch.load(
            weight_file, map_location=torch.device("cpu"), weights_only=False
        )

    model.eval()
    print("args", args)
    seq_len = 256
    while True:
        prompt = input(
            f"{''.join(['*']*50)}\nType EXIT to end the chat\n{''.join(['*']*50)}\n\nEnter prompt:"
        )
        if prompt.upper() == "EXIT":
            break
        encoded_promt = [data_loader.vocab[char] for char in prompt]
        encoded_promt_tensor = torch.tensor(encoded_promt)
        init_chars = (
            torch.nn.functional.one_hot(
                encoded_promt_tensor, num_classes=data_loader.vocab_size
            )
            .unsqueeze(0)
            .type(torch.float32)
        )
        # print(init_chars.shape)
        sequences = model.sample(init_chars, seq_len)
        sequences = torch.argmax(sequences, -1)
        seq_conv = convert(sequences, data_loader.inv_vocab)
        print()
        print("".join(seq_conv[0]))
        print("\n\n")