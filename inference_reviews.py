from tqdm import tqdm
from glob import glob
import torch
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '2'
import json
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import ElectraForSequenceClassification, ElectraTokenizer
import collections
import string

punc = string.punctuation

# load stop words
stop_words = set()
with open('stop_words.txt') as f:
    for line in f:
        stop_words.add(line.strip())
stop_words.add('[CLS]')
stop_words.add('[SEP]')
stop_words.add('’')
stop_words.add('“')
stop_words.add('"')
stop_words.add('”')
stop_words.add('…')
stop_words.add('–')
stop_words.add('story')
stop_words.add('anime')
stop_words.add('animes')
stop_words.add('episode')
stop_words.add('episodes')
stop_words.add('characters')
stop_words.add('character')


def combine_indexs(idxs, gap=3, max_length=4):
    res = []
    for idx in idxs:
        if len(res) == 0:
            res.append([idx, idx])
        elif len(res[-1]) < max_length and idx - res[-1][-1] < gap:
            res[-1][-1] = idx
        else:
            res.append([idx, idx])
    return res


def combine_tokens(tokens, idx1, idx2):
    while idx1 > 0 and tokens[idx1].startswith('##'):
        idx1 -= 1
    while idx2 < len(tokens) - 1 and tokens[idx2 + 1].startswith('##'):
        idx2 += 1
    if tokens[idx1] == '[CLS]':
        idx1 += 1
    if tokens[idx2] == '[SEP]':
        idx2 -= 1
    res = tokens[idx1:idx2 + 1]
    res = " ".join(res)
    res = res.replace(" ##", '').replace("## ", '').replace("##", '')
    res = res.strip('‘\',. ')
    return res


def is_valid_token(token):
    if token.strip() in stop_words:
        return False
    is_punc = True
    for t in token:
        if t not in punc:
            is_punc = False
            break
    if is_punc:
        return False
    return True


class ReviewDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.tokenizer = tokenizer

    def __getitem__(self, idx):
        item = tokenizer(self.data[idx], max_length=512, truncation=True, padding=True, return_tensors="pt")
        length = item['input_ids'].shape[1]
        if length < 512:
            padding_tensor = torch.zeros((1, 512 - length), dtype=item['input_ids'].dtype)
            item = {k: torch.cat([v, padding_tensor], dim=1) for k, v in item.items()}

        for k in item:
            item[k] = item[k].squeeze(0)
        return item

    def __len__(self):
        return len(self.data)


topk = 40
os.makedirs('archive/reviews_outputs', exist_ok=True)
tokenizer = ElectraTokenizer.from_pretrained('google/electra-base-discriminator')
review_list = glob('archive/reviews/*.json')
review_list.sort(reverse=True)
model = ElectraForSequenceClassification.from_pretrained("google/electra-base-discriminator")
device = torch.device('cuda')
model.to(device)
model.eval()
try:
    from apex import amp
except ImportError:
    raise ImportError("Please install apex from https://www.github.com/nvidia/apex to run this example.")
model = amp.initialize(model, None, opt_level='O1')
output_dir = './check_points/electra-base'
model.load_state_dict(torch.load(os.path.join(output_dir, 'best_checkpoint.pth'), map_location='cpu'))

with torch.no_grad():
    for rev in tqdm(review_list):
        try:
            reviews_temp = json.load(open(rev, encoding='utf8'))['reviews']
        except:
            try:
                reviews_temp = json.load(open(rev, encoding='ISO-8859-1'))['reviews']
            except:
                continue
        if len(reviews_temp) == 0:
            continue
        reviews_temp = [r.strip('Helpful read more') for r in reviews_temp]
        review_datset = ReviewDataset(reviews_temp, tokenizer)
        review_loader = DataLoader(review_datset, batch_size=32, shuffle=False, drop_last=False)
        pidx = 0
        word_count = collections.defaultdict(int)
        output = dict()

        total_preds = []
        for batch in review_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            token_type_ids = batch['token_type_ids'].to(device)
            outputs = model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids,
                            output_attentions=True)
            logits = outputs['logits']  # [batch, 2]
            preds = torch.softmax(logits, dim=1)
            preds = preds[:, 1]
            preds = preds.cpu().tolist()
            total_preds.extend(preds)
            att_sum = torch.cat(outputs['attentions'], dim=1)
            att = torch.mean(att_sum, dim=1)  # [b,t,t]
            att_cls = att[:, 0, :]  # [b,t]
            att_sort = np.argsort(att_cls.cpu().numpy(), axis=1)[:, ::-1][:, :topk]
            for i in range(att_sort.shape[0]):
                input_ids = tokenizer(reviews_temp[pidx], max_length=512, truncation=True, padding=True)['input_ids']
                input_tokens = tokenizer.convert_ids_to_tokens(input_ids)
                att_sort_ = att_sort[i]
                att_sort_.sort()
                pos_idxs = combine_indexs(att_sort_, gap=1, max_length=3)
                for pi in pos_idxs:
                    if pi[0] >= len(input_tokens) or pi[1] >= len(input_tokens):
                        continue
                    tok = combine_tokens(input_tokens, pi[0], pi[1])
                    if is_valid_token(tok) is True:
                        word_count[tok] += 1
                pidx += 1

        if len(word_count) == 0:
            continue
        word_count = [(k, v) for k, v in word_count.items()]
        word_count.sort(key=lambda x: x[1], reverse=True)
        word_count = word_count[:100]
        output['pred_scores'] = total_preds
        output['word_count'] = word_count
        with open('archive/reviews_outputs/' + rev.split('/')[-1], 'w') as w:
            json.dump(output, w, ensure_ascii=False)
