import os

os.environ['CUDA_VISIBLE_DEVICES'] = '2'
from tqdm import tqdm
from glob import glob
import numpy as np
import random
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import ElectraForSequenceClassification, \
    ElectraTokenizer, AdamW, get_linear_schedule_with_warmup
from utils import Progbar

try:
    from apex import amp
except ImportError:
    raise ImportError("Please install apex from https://www.github.com/nvidia/apex to run this example.")

random.seed(42)


def convert_imdb_data(data_list, is_pos):
    texts = []
    labels = []
    label = 1 if is_pos else 0
    for d in tqdm(data_list, desc='load'):
        f = open(d).readlines()
        texts.extend(f)
        labels.extend([label] * len(f))
    return texts, labels


def shuffle_data(pos_data, pos_labels, neg_data, neg_labels):
    data = pos_data + neg_data
    labels = pos_labels + neg_labels
    idxs = np.arange(len(labels))
    random.shuffle(idxs)
    data = list(np.array(data)[idxs])
    labels = list(np.array(labels)[idxs])
    return data, labels


class IMDbDataset(Dataset):
    def __init__(self, data, labels, tokenizer):
        self.data = data
        self.labels = labels
        self.tokenizer = tokenizer

    def __getitem__(self, idx):
        item = tokenizer(self.data[idx], max_length=512, truncation=True, padding=True, return_tensors="pt")
        length = item['input_ids'].shape[1]
        if length < 512:
            padding_tensor = torch.zeros((1, 512 - length), dtype=item['input_ids'].dtype)
            item = {k: torch.cat([v, padding_tensor], dim=1) for k, v in item.items()}

        for k in item:
            item[k] = item[k].squeeze(0)

        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


# 读取Imdb数据
imdb_path = 'aclImdb'
train_pos_list = glob(os.path.join(imdb_path, 'train', 'pos', '*.txt'))
train_neg_list = glob(os.path.join(imdb_path, 'train', 'neg', '*.txt'))
test_pos_list = glob(os.path.join(imdb_path, 'test', 'pos', '*.txt'))
test_neg_list = glob(os.path.join(imdb_path, 'test', 'neg', '*.txt'))

train_pos_data, train_pos_labels = convert_imdb_data(train_pos_list, is_pos=True)
train_neg_data, train_neg_labels = convert_imdb_data(train_neg_list, is_pos=False)
test_pos_data, test_pos_labels = convert_imdb_data(test_pos_list, is_pos=True)
test_neg_data, test_neg_labels = convert_imdb_data(test_neg_list, is_pos=False)

train_data, train_labels = shuffle_data(train_pos_data, train_pos_labels,
                                        train_neg_data, train_neg_labels)
print('Train data:', len(train_data))

test_data, test_labels = shuffle_data(test_pos_data, test_pos_labels,
                                      test_neg_data, test_neg_labels)
# split train and validation data
val_data = test_data[:2500]
val_labels = test_labels[:2500]
print('Valid data:', len(val_data))
print('Test data:', len(test_data))

tokenizer = ElectraTokenizer.from_pretrained('google/electra-base-discriminator')
train_dataset = IMDbDataset(train_data, train_labels, tokenizer)
val_dataset = IMDbDataset(val_data, val_labels, tokenizer)
test_dataset = IMDbDataset(test_data, test_labels, tokenizer)

output_dir = './check_points/electra-base'
num_train_epochs = 3
train_batch_size = 32
eval_batch_size = 64
warmup_steps = 150
weight_decay = 0.01
learning_rate = 3e-5
fp16 = True

model = ElectraForSequenceClassification.from_pretrained("google/electra-base-discriminator")
device = torch.device('cuda')
model.to(device)
model.train()
train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True, num_workers=8, drop_last=True)
val_loader = DataLoader(val_dataset, batch_size=eval_batch_size, shuffle=False, drop_last=False, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=eval_batch_size, shuffle=False, drop_last=False, num_workers=4)

steps_per_epoch = len(train_loader)
total_steps = num_train_epochs * steps_per_epoch
print('Steps per epoch:', steps_per_epoch)
print('Total steps:', total_steps)
optim = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
sched = get_linear_schedule_with_warmup(optim, num_warmup_steps=warmup_steps,
                                        num_training_steps=total_steps)
if fp16:
    model, optim = amp.initialize(model, optim, opt_level='O1')

stateful_metrics = ['epoch', 'iter', 'lr']
iteration = 0
best_acc = 0
for epoch in range(num_train_epochs):
    progbar = Progbar(len(train_dataset), max_iters=steps_per_epoch,
                      width=20, stateful_metrics=stateful_metrics)
    model.train()
    for batch in train_loader:
        optim.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        token_type_ids = batch['token_type_ids'].to(device)
        labels = batch['labels'].to(device)
        outputs = model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, labels=labels)
        loss = outputs['loss']
        if fp16:
            with amp.scale_loss(loss, optim) as loss_scaled:
                loss_scaled.backward()
        else:
            loss.backward()
        optim.step()
        sched.step()
        iteration += 1
        logs = [("epoch", epoch + 1), ("iter", iteration), ('lr', sched.get_lr()[0]), ('loss', loss.item())]
        progbar.add(train_batch_size, values=logs)

    print('\n')
    # validate
    model.eval()
    with torch.no_grad():
        total_error = 0
        for batch in tqdm(val_loader, desc='validating'):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            token_type_ids = batch['token_type_ids'].to(device)
            outputs = model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
            logits = outputs['logits']  # [batch, 2]
            preds = torch.argmax(logits, dim=1)  # [batch,]
            labels = batch['labels'].to(device)  # [batch,]
            error = torch.sum(torch.abs(preds - labels)).item()
            total_error += error
        acc = 1 - total_error / len(val_dataset)
        print('Val Acc:', acc)
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), os.path.join(output_dir, 'best_checkpoint.pth'))

# test
model.eval()
model.load_state_dict(torch.load(os.path.join(output_dir, 'best_checkpoint.pth'), map_location='cpu'))
with torch.no_grad():
    total_error = 0
    for batch in tqdm(test_loader, desc='testing'):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        token_type_ids = batch['token_type_ids'].to(device)
        outputs = model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        logits = outputs['logits']  # [batch, 2]
        preds = torch.argmax(logits, dim=1)  # [batch,]
        labels = batch['labels'].to(device)  # [batch,]
        error = torch.sum(torch.abs(preds - labels)).item()
        total_error += error
    acc = 1 - total_error / len(test_dataset)
    print('Test Acc:', acc)
