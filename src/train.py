"""Train your language model."""

import argparse

import torch
import torch.nn.functional as nnf
from tqdm import tqdm, trange

from attention_model import Transformer
from rnn_model import LSTMNet
from util import TextLoader, convert


def main():
    """Parse command line arguments and start the training."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/sturm_und_drang",
        help="data directory containing the input file",
    )
    parser.add_argument("--model_size", type=int, default=384)
    parser.add_argument("--batch_size", type=int, default=64, help="minibatch size")
    parser.add_argument(
        "--seq_length", type=int, default=256, help="Output sequence length"
    )
    parser.add_argument("--num_epochs", type=int, default=500, help="number of epochs")
    parser.add_argument(
        "--learning_rate", type=float, default=0.001, help="learning rate"
    )
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--model", default="Attention", choices=["Attention", "LSTM"])

    args = parser.parse_args()
    train(args)


def train(args):
    """Train the model."""
    torch.manual_seed(42)

    if args.device == "cuda":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            print("CUDA not available, falling back to CPU")
            device = torch.device("cpu")
    elif args.device == "mps":
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            print("MPS not available, falling back to CPU")
            device = torch.device("cpu")
    else:
        device = torch.device("cpu")
    
    if "mahabharata" in args.data_dir:
        args.batch_size = 128
        args.num_epochs = 100

    data_loader = TextLoader(args.data_dir, args.batch_size, args.seq_length)

    if args.model == "LSTM":
        model = LSTMNet(data_loader.vocab_size, args.model_size).to(device)
    else:
        model = Transformer(
            data_loader.vocab_size, args.model_size, args.seq_length, 0.2, 6
        ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    loss_fun = torch.nn.CrossEntropyLoss()

    def save():
        with open("./saved/net_save.pkl", "wb") as net_file:
            torch.save([model, args, data_loader], net_file)

    for e in trange(args.num_epochs, unit="Epoch", desc="Training Language Model"):
        data_loader.reset_batch_pointer()

        progress_bar = trange(
            data_loader.num_batches, desc="Current Epoch", leave=False
        )
        for _ in progress_bar:
            x, y = data_loader.next_batch()
            x = torch.tensor(x).to(device)
            y = torch.tensor(y).to(device)

            x_in = nnf.one_hot(x, num_classes=data_loader.vocab_size).type(
                torch.float32
            )
            model.train()
            y_hat = model(x_in)

            ce = loss_fun(y_hat.transpose(-2, -1), y)

            ce.backward()
            optimizer.step()
            optimizer.zero_grad()
            progress_bar.set_description(
                f"Current Epoch {e}. Loss {ce.detach().cpu().numpy():2.4f}"
            )

        if e % 5 == 0:
            save()
            model.eval()
            out = model(
                nnf.one_hot(x, num_classes=data_loader.vocab_size)
                .type(torch.float32)
                .to(device)
            )
            sequences = torch.argmax(out, -1)
            seq_conv = convert(sequences, data_loader.inv_vocab)
            seq_gt = convert(y, data_loader.inv_vocab)
            tqdm.write(f"Epoch: {e}. Loss: {ce.detach().cpu().numpy():2.4f}")
            tqdm.write(
                f"Epoch: {e}. Net  : " + "".join(seq_conv[0]).replace("\n", "\\n ")
            )
            tqdm.write(
                f"Epoch: {e}. Truth: " + "".join(seq_gt[0]).replace("\n", "\\n ")
            )

        if e == 50:
            tqdm.write("reducing learning rate.")
            for g in optimizer.param_groups:
                g["lr"] = g["lr"] / 2.0

    save()
    print("model saved.")


if __name__ == "__main__":
    main()
