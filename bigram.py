# %%
# importing libraries
import torch
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
num_emb = 32

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

    ix = torch.randint(len(data) - block_size, (batch_size,))
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

    def __init__(self):
        super().__init__()
        self.block_size = block_size
        self.token_embedding_table = nn.Embedding(vocab_size, num_emb)
        self.position_embedding_table = nn.Embedding(block_size, num_emb)
        self.lm_head = nn.Linear(num_emb, vocab_size)

    def forward(self, inputs, targets=None):
        B,T = inputs.shape

        token_emb = self.token_embedding_table(inputs) # batch, time, channel
        pos_emb = self.position_embedding_table(torch.arange(T, device=inputs.device))

        T = min(T, self.block_size)
        x = token_emb + pos_emb

        logits = self.lm_head(x)

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
            inputs_cropped = inputs[:, -self.block_size:]
            logits, loss = self(inputs_cropped)
            logits = logits[:, -1, :]
            prob = F.softmax(logits, 1)
            inputs_next = torch.multinomial(prob, 1)
            inputs = torch.cat((inputs, inputs_next), 1)

        return inputs

# %%
# Initialize and move the model to the correct device
model = BigramLM().to(device)

logits, loss = model(xb.to(device), yb.to(device))

print(logits.shape)
print(loss)

# %%
# making a pytorch optimizer object
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

# %%
# increasing batch size and setting a loop to evaluate loss
batch_size = 32

for i in range(max_iters):

    if i % 100 == 0:
        # losses = estimate()
        print(f"Step {i}: Loss = {loss.item()}")

    xb, yb = get_batch("train")
    logits, loss = model(xb.to(device), yb.to(device))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()



print(loss.item())

# %%
input = torch.zeros((1, 1), dtype=torch.long).to(device)
print(decode(model.generate(input, 300)[0].tolist()))

# %%
