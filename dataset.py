from torch.utils.data import Dataset, random_split, DataLoader
from constants import *
import pickle
import random
import torch
import os
import json
import numpy as np


def load_dictionary(args):
    root = os.path.join(args.root, 'dictionary/conll2003/')
    label_list = ['person', 'misc', 'organization', 'location']
    label_map = {'person': 'PER', 'location': 'LOC', 'organization': 'ORG', 'misc': 'MISC'}
    d = {}
    for l in label_list:
        with open(os.path.join(root, l + '.txt'), 'r') as f:
            words = f.read().split('\n')
        for w in words:
            if w:
                d[w] = label_map[l]
    return d


def match_dictionary(words, dictionary, n_gram=1):
    i = 0
    length = len(words)
    tags = []
    while i < length:
        is_match = False
        for n in range(1, n_gram+1):
            mention = ' '.join(words[i: i+n])
            if mention in dictionary:
                tags += [ner_label_map[dictionary[mention]]] * n
                i += n
                is_match = True
                break
        if not is_match:
            tags.append(ner_label_map['O'])
            i += 1
    return tags


def read_data(args, domain, split, unlabeled=False):
    assert split in ['train', 'valid', 'test']

    datas = []

    if args.dataset == 'ABSA':
        sp = 'test' if split == 'test' else 'train' 
        label_map = absa_label_map if args.dataset == 'ABSA' else ae_label_map
        with open(os.path.join(args.root, 'ABSA', f'{domain}_{sp}.txt'), 'r') as f:
            for line in f:
                assert '####' in line
                try:
                    words_and_tags = [(x.split('=')[0], 
                        label_map[x.split('=')[-1]] if not unlabeled else 0) 
                        for x in line.strip('\n').split('####')[-1].split()]
                    words = [item[0] for item in words_and_tags]
                    tags = [item[1] for item in words_and_tags]
                except:
                    print(line.strip('\n').split('####')[-1].split())
                datas.append((words, tags))
        if split == 'valid':
            datas = [d for i, d in enumerate(datas) if i % 10 == 0]
    
    elif args.dataset == 'AE':
        sp = 'train' if split == 'train' else 'test' 
        path = os.path.join(args.root, 'ABSA_bridge', f'{domain[:-1]}/{sp}{domain[-1]}')
        words = []
        tags = []
        with open(os.path.join(path, 'sentence.txt')) as f:
            for line in f:
                words.append(line.strip().split(' '))
        with open(os.path.join(path, 'aspect.txt')) as f:
            for line in f:
                tag = line.strip().split(' ')
                tag = [0 if int(t) == 0 else 1 for t in tag]
                # tag = [int(t) for t in tag]
                tags.append(tag)
        for word, tag in zip(words, tags):
            assert len(word) == len(tag)
            datas.append((word, tag))

    elif args.dataset == 'NER':
        if domain == 'CoNLL':
            with open(os.path.join(args.root, f'CoNLL2003/{split}.txt'), 'r') as f:
                doc = []
                for line in f:
                    line = line.strip('\n')
                    if line == '-DOCSTART- -X- -X- O':
                        continue
                    if line == '':
                        if doc == []:
                            continue
                        words = [item.split(' ')[0] for item in doc]
                        tags = [ner_label_map[item.split(' ')[-1].split('-')[-1]] for item in doc]
                        datas.append((words, tags))
                        doc = []
                    else:
                        doc.append(line)
        elif 'CBS' in domain:
            if split == 'train':
                # CBS-train is unlabeled
                if domain == 'CBS':
                    with open(os.path.join(args.root, 'CBS/large_news_tech.train'), 'r') as f:
                        for line in f:
                            words = line.strip('\n').split(' ')
                            tags = [0] * len(words)
                            datas.append((words, tags))
                # CBS labeled -> using dict
                elif domain == 'CBS_labeled':
                    dictionary = load_dictionary(args)
                    with open(os.path.join(args.root, 'CBS/large_news_tech.train'), 'r') as f:
                        for line in f:
                            words = line.strip('\n').split(' ')
                            tags = match_dictionary(words, dictionary)
                            datas.append((words, tags))
                else:
                    raise Exception(f'Undefined domain: {args.domain}!')
            else:
                dictionary = load_dictionary(args)
                tp, total_pred, total = 0, 0, 0
                with open(os.path.join(args.root, 'CBS/tech_test'), 'r', encoding='utf-8') as f:
                    sents = f.read().split('\n\n')
                    for sent in sents:
                        words_and_tags = [(x.split(' ')[0], 
                            ner_label_map[x.split(' ')[1].split('-')[-1]]) 
                            for x in sent.strip('\n').split('\n') if x and x != ' ' and not repr(x).startswith('\'\\x')]
                        words = [item[0] for item in words_and_tags]
                        tags = [item[1] for item in words_and_tags]
                        fake_tags = match_dictionary(words, dictionary)
                        for t, ft in zip(tags, fake_tags):
                            if t and ft and t == ft:
                                tp += 1
                            if ft:
                                total_pred += 1
                            # if ft and ft != t:
                            #     print(list(zip(words, tags)))
                            #     print(list(zip(words, fake_tags)))
                            #     print('-' * 50)
                            if t:
                                total += 1
                        datas.append((words, tags))
                print(tp/total_pred, tp/total)
                # exit()
        else:
            raise Exception(f'Undefined domain: {args.domain}!')
    elif args.dataset == 'QA':
        if split == 'valid':
            split = 'test'
        with open(os.path.join(args.root, domain, f'{domain}.{split}.jsonl'), 'r') as f:
            for line in f:
                json_obj = json.loads(line.strip())
                if 'header' in json_obj:
                    continue
                context_tokens = [token for token, p in json_obj['context_tokens']]
                for qas in json_obj['qas']:
                    question_tokens = [token for token, p in qas['question_tokens']]
                    answer_spans = [a['token_spans'] for a in qas['detected_answers']]
                    answer_spans = [a for item in answer_spans for a in item]
                    tags = [0] * len(context_tokens)
                    for span in answer_spans:
                        for i in range(span[0], span[1] + 1):
                            if i == span[0]:
                                tags[i] = 2
                            else:
                                tags[i] = 1
                    words = question_tokens + ['[SEP]'] + context_tokens
                    tags = [0] * (len(question_tokens) + 1) + tags
                    assert len(words) == len(tags)
                    datas.append((words, tags))
    else:
        raise Exception(f'Undefined dataset: {args.dataset}!')

    return datas


def encode_data(words_and_tags, tokenizer, max_len):
    words, tags = words_and_tags
    input_ids, masks, labels, pos_ids = [], [], [], []
    for idx, (word, tag) in enumerate(zip(words, tags)):
        if word == '[SEP]':
            sub_words = [sep_token]
        else:
            sub_words = tokenizer.encode(word, add_special_tokens=False)
        input_ids += sub_words
        masks += [1] * len(sub_words)
        labels += [tag] * len(sub_words)
        pos_ids += [idx] * len(sub_words)
    input_ids = [cls_token] + input_ids
    masks = [1] + masks
    labels = [0] + labels
    pos_ids = [-1] + pos_ids
    seq_len = len(input_ids)
    if seq_len >= max_len-1:
        input_ids = input_ids[:max_len-1] + [sep_token]
        masks = masks[:max_len-1] + [1]
        labels = labels[:max_len-1] + [0]
        pos_ids = pos_ids[:max_len-1] + [-1]
    else:
        input_ids += [sep_token] + [pad_token] * (max_len-1-seq_len)
        masks += [1] + [0] * (max_len-1-seq_len)
        labels += [0] + [0] * (max_len-1-seq_len)
        pos_ids += [-1] + [-1] * (max_len-1-seq_len)
    label_sets = set(labels)
    label_sets.remove(0)
    label_sets = list(label_sets)
    # print(label_sets)
    if len(label_sets) >= 2 or len(label_sets) == 0:
        senti_label = 0
    else:
        senti_label = label_sets[0]
    # print(labels, senti_label)
    return {
        'input_ids': np.array(input_ids),
        'masks': np.array(masks),
        'labels': np.array(labels),
        'pos_ids': np.array(pos_ids),
        'senti_label': senti_label
    }


def lstm_encode_data(words_and_tags, tokenizer, max_len):
    words, tags = words_and_tags
    input_ids, masks, labels, pos_ids = [], [], [], []
    input_ids = tokenizer.encode(words)
    masks = [1] * len(words)
    labels = tags
    pos_ids = list(range(len(words)))

    seq_len = len(input_ids)
    if seq_len > max_len:
        input_ids = input_ids[:max_len]
        masks = masks[:max_len]
        labels = labels[:max_len]
        pos_ids = pos_ids[:max_len]
    else:
        input_ids += [0] * (max_len-seq_len)
        masks += [0] * (max_len-seq_len)
        labels += [0] * (max_len-seq_len)
        pos_ids += [-1] * (max_len-seq_len)

    label_sets = set(labels)
    label_sets.remove(0)
    label_sets = list(label_sets)
    # print(label_sets)
    if len(label_sets) >= 2 or len(label_sets) == 0:
        senti_label = 0
    else:
        senti_label = label_sets[0]
    # print(labels, senti_label)
    return {
        'input_ids': np.array(input_ids),
        'masks': np.array(masks),
        'labels': np.array(labels),
        'pos_ids': np.array(pos_ids),
        'senti_label': senti_label
    }


def make_dataloader(args, tokenizer):
    dataset_factory = {
        'baseline': Single_Dataset,
        'baseline_senti_cls': Single_Dataset,
        'semi': DA_Train_Dataset,
        'DA': DA_Train_Dataset,
        'ner_dict': DA_Train_Dataset,
        'lstm_DA': LSTM_DA_Train_Dataset
    }
    train_set = dataset_factory[args.model_type](args, tokenizer, args.source, 'train')
    if args.model_type != 'lstm_DA':
        valid_set = Single_Dataset(args, tokenizer, args.source, 'valid') 
        test_set = Single_Dataset(args, tokenizer, args.target, 'test')
    else:
        valid_set = LSTM_Single_Dataset(args, tokenizer, args.source, 'valid') 
        test_set = LSTM_Single_Dataset(args, tokenizer, args.target, 'test')
    try:
        print(f'src: {train_set.src_len} tgt: {train_set.tgt_len} test: {len(test_set)}')
    except:
        print(f'train: {len(train_set)} test: {len(test_set)}')
    print(f'valid: {len(valid_set)}')
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, 
        num_workers=args.n_workers, drop_last=False)
    test_loader = DataLoader(test_set, batch_size=args.test_batch_size, shuffle=False, 
        num_workers=args.n_workers, drop_last=False)
    return train_loader, test_loader


class Single_Dataset(Dataset):
    def __init__(self, args, tokenizer, domain, split, unlabeled=False):
        self.args = args
        self.tokenizer = tokenizer
        self.split = split
        self.domain = domain
        self.data = read_data(args, self.domain, self.split, unlabeled)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return encode_data(self.data[idx], self.tokenizer, self.args.max_sent_length)


class DA_Train_Dataset(Dataset):
    def __init__(self, args, tokenizer, domain=None, split=None, unlabeled=False):
        self.args = args
        self.tokenizer = tokenizer
        self.src_data = read_data(args, args.source, 'train')
        self.tgt_data = read_data(args, args.target, 'train', unlabeled=True)
        self.src_len = len(self.src_data)
        self.tgt_len = len(self.tgt_data)

    def __len__(self):
        return self.src_len

    def __getitem__(self, idx):
        # if self.src_len < self.tgt_len:
        #     s_idx, t_idx = idx * self.src_len // self.tgt_len, idx
        # else:
        #     t_idx, s_idx = idx * self.src_len // self.tgt_len, idx
        s_idx, t_idx = idx, idx * self.tgt_len // self.src_len
        return encode_data(self.src_data[s_idx], self.tokenizer, self.args.max_sent_length), \
            encode_data(self.tgt_data[t_idx], self.tokenizer, self.args.max_sent_length)


class LSTM_Single_Dataset(Dataset):
    def __init__(self, args, tokenizer, domain, split, unlabeled=False):
        self.args = args
        self.tokenizer = tokenizer
        self.split = split
        self.domain = domain
        self.data = read_data(args, self.domain, self.split, unlabeled)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return lstm_encode_data(self.data[idx], self.tokenizer, self.args.max_sent_length)



class LSTM_DA_Train_Dataset(Dataset):
    def __init__(self, args, tokenizer, domain=None, split=None, unlabeled=False):
        self.args = args
        self.tokenizer = tokenizer
        self.src_data = read_data(args, args.source, 'train')
        self.tgt_data = read_data(args, args.target, 'train', unlabeled=True)
        self.src_len = len(self.src_data)
        self.tgt_len = len(self.tgt_data)

    def __len__(self):
        return self.src_len

    def __getitem__(self, idx):
        # if self.src_len < self.tgt_len:
        #     s_idx, t_idx = idx * self.src_len // self.tgt_len, idx
        # else:
        #     t_idx, s_idx = idx * self.src_len // self.tgt_len, idx
        s_idx, t_idx = idx, idx * self.tgt_len // self.src_len
        return lstm_encode_data(self.src_data[s_idx], self.tokenizer, self.args.max_sent_length), \
            lstm_encode_data(self.tgt_data[t_idx], self.tokenizer, self.args.max_sent_length)

