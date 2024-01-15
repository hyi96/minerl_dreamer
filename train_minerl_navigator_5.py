import numpy as np
import tqdm
import torch
import torch.optim as optim
from x_transformers import XTransformer
from torch.utils.data import DataLoader
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# learning constants
LEARNING_RATE = 5e-6
BATCH_SIZE = 8 
GRADIENT_ACCUMULATE_EVERY = 4 
PRINT_TRAIN_LOSS_EVERY = 8 # print and add to loss data every 8 gradient accumulations
NUM_TRAIN_BATCHES = GRADIENT_ACCUMULATE_EVERY * PRINT_TRAIN_LOSS_EVERY 
TEST_EVERY = PRINT_TRAIN_LOSS_EVERY * 16  
NUM_TEST_BATCHES = 16
SAVE_LOSS_EVERY = 1024 
SAVE_MODEL_EVERY = 4096

# model constants
NUM_VISION_TOKENS = 256
ENC_SEQ_LEN = 32 # action memory
DEC_SEQ_LEN = 513 # 256 vision tokens + 1 separator + 256 vision tokens
MODEL_NAME = 'minerl_navigator_5_epoch_4'


# instantiate a new model
model = XTransformer(
    dim = 768,
    enc_max_seq_len = ENC_SEQ_LEN,
    dec_max_seq_len = DEC_SEQ_LEN,
    enc_num_tokens=106,
    dec_num_tokens=258,
    enc_depth = 4,
    enc_heads = 4,
    dec_depth = 12,
    dec_heads = 12,

    num_memory_tokens = 32,
    macaron = True,
    attn_flash = True,
    pre_norm = False,       # in the paper, residual attention had best results with post-layernorm
    residual_attn = True,    # add residual attention
    use_simple_rmsnorm = True,
    ff_swish = True, # set this to True
    ff_glu = True,   # set to true to use for all feedforwards
).to(device)

# load from an existing model to run more epochs
existing_model_name = 'minerl_navigator_5_epoch_3'
model = torch.load(f'saved_models/{existing_model_name}.pth').to(device)
print(f'model {existing_model_name} loaded')


# check model size
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f'model name: {MODEL_NAME}')
print(f'total trainable parameters: {total_params}')


def cycle(loader):
    while True:
        for data in loader:
            yield data

# data loader
class Dataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data
        self.src_mask = torch.ones((ENC_SEQ_LEN,)).bool()
    def __len__(self):
        return len(self.data)
    def __getitem__(self, index):
        sample = self.data[index].long()
        src = sample[:ENC_SEQ_LEN]
        tgt = sample[ENC_SEQ_LEN:]+2 #leave space for pad token 0 and separator 1 
        tgt[NUM_VISION_TOKENS] = 1 # separator
        return src, tgt, self.src_mask

def preprocess_data(data):
    return torch.cat((data[:,:ENC_SEQ_LEN+NUM_VISION_TOKENS], torch.zeros((data.size()[0],1), dtype=torch.uint8), data[:,-NUM_VISION_TOKENS:]), 1)


# load and preprocess data
print('loading datasets...')
train_data = preprocess_data(torch.load('data/minerl_train.pt')) # K samples
test_data = preprocess_data(torch.load('data/minerl_test.pt')) # 50K samples
print('training data size:', train_data.size())
print('testing data size:', test_data.size())
train_data = DataLoader(Dataset(train_data), batch_size=BATCH_SIZE, num_workers=8, pin_memory=True, shuffle=True)
test_data = cycle(DataLoader(Dataset(test_data), batch_size=BATCH_SIZE, num_workers=8, pin_memory=True, shuffle=True))


# train model
optim = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=0.001)

train_loss_data = []
test_loss_data = []
train_loss_save_path = f'loss_data/{MODEL_NAME}_train_loss.npy'
test_loss_save_path = f'loss_data/{MODEL_NAME}_test_loss.npy'
train_loss_acum = 0
grad_accum_counter = 0
accum_batch_counter = 0
model.train()
for i, (src, tgt, mask) in enumerate(tqdm.tqdm(train_data, desc="training")):
    src, tgt, mask = src.to(device), tgt.to(device), mask.to(device)
    loss = model(src, tgt, mask=mask)
    train_loss_acum += loss.item()
    (loss/GRADIENT_ACCUMULATE_EVERY).backward()
    grad_accum_counter += 1
    if grad_accum_counter == GRADIENT_ACCUMULATE_EVERY:
            optim.step()
            optim.zero_grad()
            grad_accum_counter = 0
            accum_batch_counter += 1
            
            if accum_batch_counter % PRINT_TRAIN_LOSS_EVERY == 0:
                train_loss_acum /= NUM_TRAIN_BATCHES
                train_loss_data.append(train_loss_acum)
                print(f'iter={accum_batch_counter} loss={train_loss_acum}')
                train_loss_acum = 0
            if accum_batch_counter % TEST_EVERY == 0:
                model.eval()
                test_loss_acum = 0
                for _ in range(NUM_TEST_BATCHES):
                    src, tgt, mask = next(test_data)
                    src, tgt, mask = src.to(device), tgt.to(device), mask.to(device)
                    test_loss_acum += model(src, tgt, mask=mask).item()
                test_loss_acum /= NUM_TEST_BATCHES
                test_loss_data.append(test_loss_acum)
                print(f'iter={accum_batch_counter} test loss={test_loss_acum}')
                model.train()
            if accum_batch_counter % SAVE_LOSS_EVERY == 0: 
                np.save(train_loss_save_path, train_loss_data) 
                np.save(test_loss_save_path, test_loss_data)
            if accum_batch_counter % SAVE_MODEL_EVERY == 0: 
                torch.save(model, f'saved_models/{MODEL_NAME}.pth')
                print(f'model {MODEL_NAME} saved')

print('total accumulated batches:', accum_batch_counter)
# save model
torch.save(model, f'saved_models/{MODEL_NAME}.pth')
print(f'model {MODEL_NAME} saved')
np.save(train_loss_save_path, train_loss_data) 
np.save(test_loss_save_path, test_loss_data)

