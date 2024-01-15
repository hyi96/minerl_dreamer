import tqdm
import torch
import torch.optim as optim
from x_transformers import TransformerWrapper, Encoder
import random
import torch.nn as nn

# constants
NUM_BATCHES = int(1e5)
BATCH_SIZE = 32
LEARNING_RATE = 1e-5
PRINT_LOSS_EVERY = 50
EVAL_EVERY  = 500
NUM_TOKENS = 10
ENC_SEQ_LEN = 10
DEC_SEQ_LEN = 1


# instantiate a new model
model = TransformerWrapper(
    num_tokens = NUM_TOKENS,
    max_seq_len = ENC_SEQ_LEN,
    attn_layers = Encoder(
        dim = 768,
        depth = 24,
        heads = 12,
        attn_flash=True,
    )
).cuda()


#load data
with open('../data/rule30-midcolumn-a-million-bits.txt', 'r') as file: data = file.read()
data = data.strip()
data = list(map(int, list(data)))
num_samples = len(data)
def encode_input(inp):
    encoded = list(map(int, list(str(inp))))
    padding = ENC_SEQ_LEN - len(encoded)
    if padding<0:
        raise Exception("input exceeds ENC_SEQ_LEN")
    else:
        return [0]*padding + encoded


data = [encode_input(i) + [data[i]] for i in range(len(data))]
data = torch.tensor(data)
print('data size:',data.size())

test_data = data[int(len(data)*0.9):] #last 10% for testing
train_data = data[:int(len(data)*0.9)]


# data batch loader
def cycle(is_train=True):
    data = train_data if is_train else test_data
    while True:
        i = random.randint(0,len(data)-BATCH_SIZE-1)
        batch = data[i:i+BATCH_SIZE]
        src = batch[:,:ENC_SEQ_LEN]
        tgt = batch[:,ENC_SEQ_LEN:]
        yield (src, tgt)



total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f'Total trainable parameters: {total_params}')

# optimizer
optim = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

print('start training')
total_eval = 1
total_correct = 0
criterion = nn.CrossEntropyLoss()
# training
for i in tqdm.tqdm(range(NUM_BATCHES), mininterval=10., desc='training'):
    model.train()
    src, tgt = next(cycle())
    src, tgt = src.cuda(), tgt.cuda()
    optim.zero_grad()
    logits = model(src)
    loss = criterion(logits[:,:DEC_SEQ_LEN].permute(0,2,1), tgt)
    loss.backward()
    optim.step()
    if i % PRINT_LOSS_EVERY == 0: print(f'iter={i} loss={loss.item()}')
    if i % EVAL_EVERY == 0:
        model.eval()
        src, tgt = next(cycle(False))
        src, tgt = src.cuda(), tgt.cuda()
        with torch.no_grad(): 
            logits = model(src)
            loss = criterion(logits[:,:DEC_SEQ_LEN].permute(0,2,1), tgt)
            print(f'iter={i} test loss={loss.item()}')

torch.save(model, '../saved_models/rule30_mid_column_decoder.pth')