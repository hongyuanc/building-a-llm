# %%
# importing libraries
import torch
import tiktoken
import torch.nn as nn
from torch.nn import functional as F
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

# %%
# read the text file
def read_text_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()
    return text

# %%
# feeding the text and cleaning it
file_path = 'training_text.txt'
text = read_text_file(file_path)

text = text.strip("\ufeff")
text = text.replace("\n", " ")

# %%
# length of text
print(len(text))
print(text[:100])

# %%
# checking all the characters used in the text
char = sorted(list(set(text)))
vocab_size = len(char)
print(''.join(char))
print(vocab_size)
eval_iters = 200
max_iters = 10000

# %%
# bringing in tiktoken's tokenizer

# THIS WILL NOT WORK

enc = tiktoken.get_encoding("gpt2")
enc.n_vocab

# %%
# testing with encoding and decoding; it works now, but note that we only have
# a vocab_size of 85
encode = enc.encode("hello")
print(encode)
enc.decode(encode)

encode1 = enc.encode(text[:100])
print(encode1)
enc.decode(encode1)

# %%
# hence we need to tokenize the vocab ourselves
stoi = { ch:i for i, ch in enumerate(char)}
itos = { i:ch for i, ch in enumerate(char)}

def encode(str):
    return [stoi[c] for c in str]

def decode(data):
    return "".join(itos[c] for c in data)

foo = encode("foo")
print(foo)
print(decode(foo))

# %%
# testing encode with my own functions
# notice the difference between tiktoken
# but we cannot use tiktoken so this will do
test_encode = encode(text[:100])
print(test_encode)
print(decode(test_encode))

# %%
# now encoding the entire text
data = torch.tensor(encode(text), dtype=torch.long)
print(data.shape, data.dtype)
print(data[:500])

# %%
# splitting data into training data and validation data
n = int(0.9*len(data))
train = data[:n]
val = data[n:]

# %%
# setting up block size
block_size = 8
train[:block_size + 1]

# %%
# setting batch size
batch_size = 4

# function for getting a batch of random blocks within data, set my batch_size
def get_batch(split):
    if split == "train":
        data = train
    else:
        data = val

    ix = torch.randint(len(data) - batch_size, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1: i + block_size + 1] for i in ix])
    
    return x, y

# estimate function that estimates the average loss in splits
def estimate():
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for i in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[i] = loss.item()
        out[split] = losses.mean() # collecting the average
    model.train()

    return out


# %%
# collecting inputs and targets from the training data
# targets used for creating the loss function later on
xb, yb = get_batch("train")

print(xb)
print(yb)

# %%
# class for the language model
class BigramLM(nn.Module):

    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, inputs, targets=None):
        logits = self.token_embedding_table(inputs) # batch, time, channel

        if targets is None:
            loss = None
        else:
            # need to reformat BTC into B*C, T for loss to work
            b, t, c = logits.shape
            logits = logits.view(b*t, c)

            # targets are in B T and needs to be B*T
            targets = targets.view(b*t)

            loss = F.cross_entropy(logits, targets)

        return logits, loss
    
    def generate(self, inputs, number):
        for _ in range(number):
            logits, loss = self(inputs)
            logits = logits[:, -1, :]
            prob = F.softmax(logits, 1)
            inputs_next = torch.multinomial(prob, 1)
            inputs = torch.cat((inputs, inputs_next), 1)

        return inputs
    

# %% [markdown]
# Usually we would expect a loss of -ln(1/86), which is approximately **-4.45**, 
# but we are getting almost 5 right now. This means the inital predictions are not very diffused yet, and there is entropy.

# %%
# this was where i found that tiktoken wouldn't work
model = BigramLM(vocab_size)

logits, loss = model(xb, yb)

print(logits.shape)
print(loss)

# %% [markdown]
# This looks silly now because history is not used; we only examine the last character in time: **logits = logits[:, -1, :]**

# %%
# setting an input as a 1 by 1 tensor of zeros
input = torch.zeros((1, 1), dtype=torch.long)
print(decode(model.generate(input, 100)[0].tolist()))

# %%
# making a pytorch optimzer object
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

# %% [markdown]
# **Training the model.**
# We can see the loss going down everytime we train it.

# %%
# increasing batch size and setting a loop to evaluate loss
batch_size = 32

for i in range(max_iters): # my computer almost blew up

    losses = estimate()

    xb, yb = get_batch("train")
    logits, loss = model(xb, yb)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

print(loss.item())
print(f"step {i}: train loss: {losses['train']:.4f}, val loss {losses['val']:.4f}")

# %% [markdown]
# As we can see, there are drastic improvements than to before.
# 
# 
# "—;84c7.nd]ontxe$4neHLKyéxUa—fW]’g0IySCPjoM3j,4R 0h—$é ],Oé4x$érBXJHP12MGay8?W’O[ ?[3xIV9é?*NC “va1G" 
# 
# to
# 
# 
# " d I wnt harsurthed is owemeveg wayouio id aly a this wicy’se  d Minghe Shouad yoofezzive rnan y dineng o 
# thiniss mecegs wayth inn r  “Antoke!” s ge, motthof, ay wes blut cemomos. astr I wa dad Wes dyecakneitlastat ceche. “
# Ale nidosef My I roufol amed cousy ay?”  s, od. son’tanonghat athad Routheed d".

# %%
input = torch.zeros((1, 1), dtype=torch.long)
print(decode(model.generate(input, 300)[0].tolist()))


