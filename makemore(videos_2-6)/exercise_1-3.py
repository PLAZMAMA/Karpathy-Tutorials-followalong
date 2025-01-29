import math
from code import interact
from copy import deepcopy
from itertools import permutations
from random import shuffle

import torch
import torch.nn.functional as torch_func


def train_dev_test_split(
    words: list[str], train_percentage: float = 0.8, dev_percentage: float = 0.1
) -> tuple[list[str], list[str], list[str]]:
    shuffled_words = deepcopy(words)
    shuffle(shuffled_words)
    train_dev_split_point = math.ceil(len(words) * train_percentage)
    dev_val_split_point = math.floor(
        train_dev_split_point + (len(words) * dev_percentage)
    )
    return (
        shuffled_words[:train_dev_split_point],
        shuffled_words[train_dev_split_point:dev_val_split_point],
        shuffled_words[dev_val_split_point:],
    )


def create_bigram_dataset(
    words: list[str], char_to_indx: dict[str, int]
) -> tuple[torch.Tensor, list[int]]:
    xs = []
    ys: list[int] = []
    for word in words:
        word_chars = (
            ["."] + list(word) + ["."]
        )  # Adding starting and ending chars which '.' represents, ex. "cat" -> ".cat."

        # Creating inputs and labels from the current word
        for char1, char2 in zip(word_chars, word_chars[1:]):
            xs.append(char_to_indx[char1])
            ys.append(char_to_indx[char2])

    enc_xs = torch_func.one_hot(torch.tensor(xs), num_classes=len(char_to_indx)).float()
    enc_xs.requires_grad_()
    return enc_xs, ys


def bigram_nn():
    # Create bigram matrix
    words = open("names.txt", "r").read().splitlines()
    chars = sorted(list(set("".join(words))))
    char_to_indx = {char: indx for indx, char in enumerate(chars)}
    char_to_indx["."] = 26
    # indx_to_char = {indx: char for char, indx in char_to_indx.items()}

    train_words, dev_words, test_words = train_dev_test_split(words)
    train_enc_xs, train_ys = create_bigram_dataset(train_words, char_to_indx)
    dev_enc_xs, dev_ys = create_bigram_dataset(dev_words, char_to_indx)
    test_enc_xs, test_ys = create_bigram_dataset(test_words, char_to_indx)

    # Init weights
    W = torch.randn((len(char_to_indx), len(char_to_indx)), requires_grad=True)

    EPOCHS = 500
    LEARNING_RATE = 20

    # Train
    for _ in range(EPOCHS):
        # Forward pass
        logits = train_enc_xs @ W  # Log counts

        # Cost funcion (Softmax)
        counts = logits.exp()  # Equivilant of a row in the Bigram marix above
        probs = counts / counts.sum(1, keepdim=True)

        # Calculating loss and printing it
        neg_log_likelihood = (
            -probs[torch.arange(probs.shape[0]), train_ys].log() + 0.01 * (W**2).mean()
        )
        loss = neg_log_likelihood.mean()
        print(f"{loss.item() = }")

        # Backpropogation
        if W.grad is not None:
            W.grad.zero_()
        loss.backward()

        # Update weights
        W.data += -LEARNING_RATE * W.grad  # pyright: ignore[reportOperatorIssue]

    eval_enc_xs = test_enc_xs
    eval_ys = dev_ys + test_ys
    # Forward pass
    logits = eval_enc_xs @ W  # Log counts

    # Cost funcion (Softmax)
    counts = logits.exp()  # Equivilant of a row in the Bigram marix above
    probs = counts / counts.sum(1, keepdim=True)

    # Calculating loss and printing it
    neg_log_likelihood = (
        -probs[torch.arange(probs.shape[0]), eval_ys].log() + 0.01 * (W**2).mean()
    )
    loss = neg_log_likelihood.mean()

    # # Sample names from the neural network. AKA inference.
    # gen = torch.Generator().manual_seed(2147483647)
    #
    # start_char_num = W.shape[0] - 1
    # end_char_num = W.shape[0] - 1
    # for _ in range(20):
    #     out = ""
    #     char_num = start_char_num
    #     while True:
    #         one_hot_enc = torch_func.one_hot(
    #             torch.tensor(char_num), num_classes=W.shape[0]
    #         ).float()
    #         logits = one_hot_enc @ W  # Log(base e) counts
    #         counts = logits.exp()
    #         probs = counts / counts.sum()
    #         char_num = torch.multinomial(
    #             probs, num_samples=1, replacement=True, generator=gen
    #         ).item()
    #         out += indx_to_char[char_num]  # pyright: ignore[reportArgumentType]
    #         if char_num == end_char_num:
    #             break
    #
    #     print(out[:-1])


def create_trigram_dataset(
    words: list[str], char_to_indx: dict[str, int], char_pair_to_indx: dict[str, int]
) -> tuple[list[int], list[int]]:
    xs: list[int] = []
    ys: list[int] = []
    for word in words:
        word_chars = ["."] + list(word) + ["."]
        for char1, char2, char3 in zip(word_chars, word_chars[1:], word_chars[2:]):
            xs.append(char_pair_to_indx[char1 + char2])
            ys.append(char_to_indx[char3])

    return xs, ys


def trigram_nn():
    words = open("names.txt", "r").read().splitlines()
    chars = sorted(list(set("".join(words))))
    char_to_indx = {char: indx for indx, char in enumerate(chars)}
    char_to_indx["."] = 26
    chars_pairs = sorted(
        set(("".join(perm) for perm in permutations(chars * 2 + ["."], 2)))
    )
    char_pair_to_indx = {pair: indx for indx, pair in enumerate(chars_pairs)}
    # indx_to_char = {indx: char for char, indx in char_to_indx.items()}

    train_words, dev_words, test_words = train_dev_test_split(words)

    train_xs, train_ys = create_trigram_dataset(
        train_words, char_to_indx, char_pair_to_indx
    )
    dev_xs, dev_ys = create_trigram_dataset(
        dev_words, char_to_indx, char_pair_to_indx
    )
    test_xs, test_ys = create_trigram_dataset(
        test_words, char_to_indx, char_pair_to_indx
    )


    EPOCHS = 500

    learning_rate = 50
    cross_entropy_loss = torch.nn.CrossEntropyLoss()

    train_xs += dev_xs
    train_ys = torch.tensor(train_ys + dev_ys)
    eval_xs = test_xs
    eval_ys = torch.tensor(test_ys)

    # Init weights
    weights = torch.randn((len(char_pair_to_indx), len(char_to_indx)), requires_grad=True)

    for current_epoch in range(EPOCHS):
        # Forward pass
        logits = weights[train_xs] # Log counts

        loss:torch.Tensor = cross_entropy_loss(logits, train_ys)
        print(f"Progress: {current_epoch}/{EPOCHS}, Loss: {loss}", end="\r")

        # Backpropogation
        if weights.grad is not None:
            weights.grad.zero_()
        loss.backward()

        # Update weights
        weights.data += -learning_rate * weights.grad  # pyright: ignore[reportOperatorIssue]

    # Forward pass
    logits = weights[eval_xs] # Log counts
    loss = cross_entropy_loss(logits, eval_ys)

    print(" " * 100, end="\r") # Clearing line
    print("learning_rate:", learning_rate, "Loss:", loss.item())

if __name__ == "__main__":
    trigram_nn()
    # print("=" * 50, "BIGRAM", "=" * 50)
    # bigram_nn()
