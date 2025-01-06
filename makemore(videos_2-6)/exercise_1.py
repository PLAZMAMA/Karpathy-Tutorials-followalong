from code import interact
from itertools import permutations
from random import randint
import torch
import torch.nn.functional as torch_func


def bigram_nn():
    # Create bigram matrix
    words = open("names.txt", "r").read().splitlines()
    chars = sorted(list(set("".join(words))))
    char_to_indx = {char: indx for indx, char in enumerate(chars)}
    char_to_indx["."] = 26
    indx_to_char = {indx: char for char, indx in char_to_indx.items()}
    bigrams = torch.zeros([len(char_to_indx)]*2, dtype=torch.int32)
    bigrams += 1 # smooth probabilities(at traning time) by making sure the minimum ammout of counts a bigram has is 1 or bigger.


    xs = []
    ys = []
    for word in words:
        word_chars = ["."] + list(word) + ["."] # Adding starting and ending chars which '.' represents, ex. "cat" -> ".cat."

        # Creating inputs and labels from the current word
        for char1, char2 in zip(word_chars, word_chars[1:]):
            xs.append(char_to_indx[char1])
            ys.append(char_to_indx[char2])

    enc_xs = torch_func.one_hot(
        torch.tensor(xs),
        num_classes = len(char_to_indx)
    ).float()
    enc_xs.requires_grad_()
    enc_ys = torch_func.one_hot(
        torch.tensor(ys),
        num_classes = len(char_to_indx)
    ).float()
    enc_ys.requires_grad_()

    # Init weights
    W = torch.randn((len(char_to_indx), len(char_to_indx)), requires_grad=True)

    EPOCHS = 200
    LEARNING_RATE = 20

    for _ in range(EPOCHS):
        # Forward pass
        logits = enc_xs @ W # Log counts

        # Cost funcion (Softmax)
        counts = logits.exp() # Equivilant of a row in the Bigram marix above
        probs = counts / counts.sum(1, keepdim=True)

        # Calculating loss and printing it
        neg_log_likelihood = -probs[torch.arange(probs.shape[0]), ys].log() + 0.01 * (W**2).mean()
        loss = neg_log_likelihood.mean()
        print(f"{loss.item() = }")

        # Backpropogation
        if W.grad is not None:
            W.grad.zero_() # pyright: ignore[reportUnusedCallResult]
        loss.backward()

        # Update weights
        W.data += -LEARNING_RATE * W.grad # pyright: ignore[reportOperatorIssue] 


    # Sample names from the neural network. AKA inference.
    gen = torch.Generator().manual_seed(2147483647)

    start_char_num = W.shape[0] - 1
    end_char_num = W.shape[0] - 1
    for _ in range(20):
        out = ""
        char_num = start_char_num
        while True:
            one_hot_enc = torch_func.one_hot(torch.tensor(char_num), num_classes=W.shape[0]).float()
            logits = one_hot_enc @ W # Log(base e) counts
            counts = logits.exp()
            probs = counts / counts.sum()
            char_num = torch.multinomial(probs, num_samples=1, replacement=True, generator=gen).item()
            out += indx_to_char[char_num] # pyright: ignore[reportArgumentType]
            if char_num == end_char_num:
                break

        print(out[:-1])

def trigram_nn():
    words = open("names.txt", "r").read().splitlines()
    chars = sorted(list(set("".join(words))))
    char_to_indx = {char: indx for indx, char in enumerate(chars)}
    char_to_indx["."] = 26
    chars_pairs = sorted(set(("".join(perm) for perm in permutations(chars*2 + ["."], 2))))
    char_pair_to_indx = {pair: indx for indx, pair in enumerate(chars_pairs)}
    indx_to_char = {indx: char for char, indx in char_to_indx.items()}

    xs: list[int] = []
    ys: list[int] = []
    for word in words:
        word_chars = ["."] + list(word) + ["."]
        for char1, char2, char3 in zip(word_chars, word_chars[1:], word_chars[2:]):
            xs.append(char_pair_to_indx[char1+char2])
            ys.append(char_to_indx[char3])

    enc_xs = torch_func.one_hot(
        torch.tensor(xs),
        num_classes = len(char_pair_to_indx)
    ).float()
    enc_xs.requires_grad_() # pyright: ignore[reportUnusedCallResult]
    enc_ys = torch_func.one_hot(
        torch.tensor(ys),
        num_classes = len(char_to_indx)
    ).float()
    enc_ys.requires_grad_() # pyright: ignore[reportUnusedCallResult]


    # Init weights
    W = torch.randn((len(char_pair_to_indx), len(char_to_indx)), requires_grad=True)

    EPOCHS = 500
    LEARNING_RATE = 20

    for _ in range(EPOCHS):
        # Forward pass
        logits = enc_xs @ W # Log counts

        # Cost funcion (Softmax)
        counts = logits.exp() # Equivilant of a row in the Bigram marix above
        probs = counts / counts.sum(1, keepdim=True)

        # Calculating loss and printing it
        neg_log_likelihood = -probs[torch.arange(probs.shape[0]), ys].log() + 0.01 * (W**2).mean()
        loss = neg_log_likelihood.mean()
        print(f"{loss.item() = }")

        # Backpropogation
        if W.grad is not None:
            W.grad.zero_() # pyright: ignore[reportUnusedCallResult]
        loss.backward() # pyright: ignore[reportUnknownMemberType, reportUnusedCallResult]

        # Update weights
        W.data += -LEARNING_RATE * W.grad # pyright: ignore[reportOperatorIssue, reportUnknownMemberType] 


    # Sample names from the neural network. AKA inference.
    SAMPLE_SIZE = 20
    gen = torch.Generator().manual_seed(2147483647)

    end_char_num = char_to_indx["."]
    for _ in range(SAMPLE_SIZE):
        in_char_pair = "." + indx_to_char[randint(0, len(indx_to_char)-2)]
        in_char_pair_num = char_pair_to_indx[in_char_pair]
        out = in_char_pair
        while True:
            one_hot_enc = torch_func.one_hot(torch.tensor(in_char_pair_num), num_classes=len(char_pair_to_indx)).float()
            logits = one_hot_enc @ W # Log(base e) counts
            counts = logits.exp()
            probs = counts / counts.sum()
            out_char_num = torch.multinomial(probs, num_samples=1, replacement=True, generator=gen).item()
            out_char = indx_to_char[out_char_num] # pyright: ignore[reportArgumentType]
            in_char_pair = in_char_pair[1] + out_char
            in_char_pair_num = char_pair_to_indx[in_char_pair]
            out += out_char
            if out_char_num == end_char_num:
                break

        print(out[1:-1])

if __name__ == "__main__":
    trigram_nn()
    print("=" *50, "BIGRAM", "="*50)
    # bigram_nn()
