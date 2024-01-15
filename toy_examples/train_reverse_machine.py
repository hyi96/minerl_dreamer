import tqdm
import torch
import torch.optim as optim
from x_transformers import XTransformer

# constants
NUM_BATCHES = int(1e4)
BATCH_SIZE = 32
LEARNING_RATE = 3e-4
PRINT_LOSS_EVERY = 50
GENERATE_EVERY  = 200
NUM_TOKENS = 16 + 2
ENC_SEQ_LEN = 32
DEC_SEQ_LEN = 32


# helpers
def cycle():
    while True:
        prefix = torch.ones((BATCH_SIZE, 1)).long().cuda()
        src = torch.randint(2, NUM_TOKENS, (BATCH_SIZE, ENC_SEQ_LEN)).long().cuda()
        tgt = torch.cat((prefix, torch.flip(src, dims=[1]).cuda()), 1)
        src_mask = torch.ones_like(src).bool().cuda()
        yield (src, tgt, src_mask)


# instantiate model
model = XTransformer(
    dim = 512,
    tie_token_emb = True,
    enc_num_tokens=NUM_TOKENS,
    enc_depth = 3,
    enc_heads = 8,
    enc_max_seq_len = ENC_SEQ_LEN,
    dec_num_tokens = NUM_TOKENS,
    dec_depth = 3,
    dec_heads = 8,
    dec_max_seq_len = DEC_SEQ_LEN
).cuda()

total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f'Total trainable parameters: {total_params}')

# optimizer
optim = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)


print('start training')
total_eval = 1
total_correct = 0
# training
for i in tqdm.tqdm(range(NUM_BATCHES), mininterval=10., desc='training'):
    model.train()
    src, tgt, src_mask = next(cycle())
    loss = model(src, tgt, mask=src_mask)
    loss.backward()

    if i % PRINT_LOSS_EVERY == 0: print(f'iter={i} loss={loss.item()}')

    optim.step()
    optim.zero_grad()

    if i != 0 and i % GENERATE_EVERY == 0:
        model.eval()
        total_eval += 1
        src, tgt, src_mask = next(cycle())
        src, src_mask, tgt = src[:1], src_mask[:1], tgt[:1]
        start_tokens = (torch.ones((1, 1)) * 1).long().cuda()
        sample = model.generate(src, start_tokens, DEC_SEQ_LEN, mask = src_mask)
        total_correct += 1 if torch.equal(sample, tgt[:,1:]) else 0
        print(f"input:  ", src)
        print(f"predicted output:  ", sample)
        print(f"expected output:  ", tgt[:,1:])
        print(f"accuracy:  ", total_correct/total_eval)

torch.save(model, '../saved_models/reverse_machine.pth')