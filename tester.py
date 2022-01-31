import torch
import numpy as np
from utils import AverageMeter
import os
from dataset import read_data
import json


def load_dictionary():
    root = './data/dictionary/conll2003/'
    list = ['person', 'misc', 'organization', 'location']
    d = {}
    for l in list:
        with open(root + l + '.txt', 'r') as f:
            words = f.read().split('\n')
        for w in words:
            if w:
                d[w] = l
    return d

def get_f1(n_hit, n_preds, n_total):
    precision = n_hit / (n_preds + 1e-6)
    recall = n_hit / (n_total + 1e-6)
    F1 = 2 * precision * recall / (precision + recall + 1e-6)
    return precision * 100, recall * 100, F1 * 100


def match(pred, label):
    pred = [(t1, t2) for t1, t2, t3 in pred]
    label = [(t1, t2) for t1, t2, t3 in label]
    cnt = 0
    for t in label:
        if t in pred:
            cnt += 1
    return cnt


def match_sentiment(pred, label):
    cnt = 0
    for t in label:
        if t in pred:
            cnt += 1
    return cnt


# 0 0 0 1 1 0 -> (3, 4)
def convert_tag(tag_seq):
    results = []
    i = 0
    cur_entity_tag = 0
    while i < len(tag_seq):
        cur_entity_tag = tag_seq[i]
        for k in range(i, len(tag_seq)+1):
            if k == len(tag_seq) or tag_seq[k] != cur_entity_tag:
                if cur_entity_tag != 0:
                    results.append((i, k-1, cur_entity_tag))
                break
        i = k   
    return results

# 0 0 0 1 2 0 -> (3, 4)
def convert_tag_BIO(tag_seq):
    results = []
    begin = -1
    for idx, tag in enumerate(tag_seq):
        if tag == 2:
            continue
        elif tag == 1:
            if begin != -1:
                results.append((begin, idx-1, 1))
            begin = idx
        else:
            if begin != -1:
                results.append((begin, idx-1, 1))
                begin = -1
    return results


def parse(probs, labels, masks, args):
    preds = torch.argmax(torch.tensor(probs), dim=-1).tolist()
    seq_len = sum(masks)
    if args.model_type != 'lstm_DA':
        preds = preds[1:seq_len-1]
        labels = labels[1:seq_len-1]
    else:
        preds = preds[:seq_len]
        labels = labels[:seq_len]
    if len(probs[0]) == 3:
        preds = convert_tag_BIO(preds)
        labels = convert_tag_BIO(labels)
    else:
        preds = convert_tag(preds)
        labels = convert_tag(labels)
    return preds, labels


class BaseTester(object):
    def __init__(self, args, model, test_loader):
        self.args = args
        self.model = model
        self.test_loader = test_loader
        self.loss_func = torch.nn.CrossEntropyLoss()
        self.model_dir = os.path.join(args.output_dir, args.output_name)

    def test_one_epoch(self, device, epoch, debug=False):
        self.infer(device, epoch, debug)
        return self.count_metrics(device, epoch, debug)

    def infer(self, device, epoch, debug=False):
        # loss_meter = AverageMeter()
        self.loss_func = self.loss_func.to(device)
        self.model.eval()
        n_correct_absa, n_correct_ae, n_preds, n_total = 0, 0, 0, 0
        pred_list, label_list, pos_list, prob_list = [], [], [], []
        features_v, labels_v = [], []
        raw_inputs = read_data(self.args, self.args.target, 'test')
        instance_cnt = 0
        infer_output_path = self.model_dir + f'_{epoch}_infer.out'
        if os.path.exists(infer_output_path):
            os.remove(infer_output_path)
        with open(infer_output_path, 'a') as f:
            with torch.no_grad():
                for j, inputs in enumerate(self.test_loader):
                    inputs = {k: inputs[k].to(device) for k in inputs}
                    logits, output, _ = self.model(input_ids=inputs['input_ids'], masks=inputs['masks'])
                    batch_size = inputs['input_ids'].shape[0]
                    for i in range(batch_size):
                        result_dict = {
                            'raw_input': raw_inputs[instance_cnt][0],
                            'pos_id': inputs['pos_ids'][i].tolist(),
                            'probs': torch.softmax(logits[i], dim=-1).tolist(),
                            'masks': inputs['masks'][i].tolist(),
                            'labels': inputs['labels'][i].tolist()
                        }
                        f.write(json.dumps(result_dict, ensure_ascii=False) + '\n')
                        instance_cnt += 1

                    # if debug:
                    #     output = output.tolist()
                    #     masks = inputs['masks'].tolist()
                    #     ls = inputs['labels'].tolist()
                    #     for l, m, f in zip(ls, masks, output):
                    #         max_l = sum(m)
                    #         for feat, label in zip(f[:max_l], l[:max_l]):
                    #             features_v.append(feat)
                    #             labels_v.append(label)
            # if debug:
            #     import pickle
            #     pickle.dump((features_v, labels_v), open('output.pkl', 'wb'))
            #     print('dump!')

    def count_metrics(self, device, epoch, debug=False):
        all_entity = set()
        n_correct_absa, n_correct_ae, n_preds, n_total = 0, 0, 0, 0
        infer_output_path = os.path.join(self.model_dir + f'_{epoch}_infer.out')
        with open(infer_output_path, 'r') as f:
            for line in f:
                obj = json.loads(line.strip())
                words = obj['raw_input']
                pred, gold = parse(obj['probs'], obj['labels'], obj['masks'], self.args)
                def to_str(begin, end, senti):
                    # print(begin, end)
                    # print(words)
                    if self.args.model_type != 'lstm_DA':
                        raw_pos = sorted(list(set([obj['pos_id'][i+1] for i in range(begin, end+1)])))
                    else:
                        raw_pos = list(range(begin, end+1))
                    entity = ' '.join([words[i] for i in raw_pos])
                    entities_with_senti = (entity, senti, min(raw_pos), max(raw_pos))
                    return entities_with_senti

                pred_entities = [to_str(x, y, z) for x, y, z in pred]
                gold_entities = [to_str(x, y, z) for x, y, z in gold]
                pred_entities_no_senti = [(entity, b, e) for entity, _, b, e in pred_entities]
                gold_entities_no_senti = [(entity, b, e) for entity, _, b, e in gold_entities]

                # if debug and not pred_entities and len(gold_entities) > 1:
                if debug:
                    assert len(words) == max(obj['pos_id']) + 1
                    assert len(gold) == len(gold_entities)
                    pos_to_prob_map = {}
                    for x, p in zip(obj['pos_id'][1:], obj['probs'][1:]):
                        if x == -1:
                            break
                        pos_to_prob_map[x] = pos_to_prob_map.get(x, []) + [p]
                    # print(pos_to_prob_map)
                    for x, word in enumerate(words):
                        pp = pos_to_prob_map[x]
                        print(f'{word}({pp})', end=' ')
                        # if np.mean(np.array(pp), axis=0)[0] > 0.2:
                        #     continue
                        # label_map = {'person': 1, 'location': 2, 'organization': 3, 'misc': 4}
                        # t = label_map[d[word]]
                        # print(t)
                        # t = np.argmax(t)
                        # print(t)
                        # pred_entities = [phrase for phrase in pred_entities if phrase[-2] != x and phrase[-1] != x]
                        # pred_entities.append((word, t, x, x))
                    print('')
                    print('pred entity: ' + ' | '.join([f'{e}({t})' for e, t, _, _ in pred_entities]))
                    print('gold entity: ' + ' | '.join([f'{e}({t})' for e, t, _, _ in gold_entities]))
                    print('-'*20)

                    # for phrase in gold_entities:
                    #     if phrase not in pred_entities:
                    #         if phrase[0] in d:
                    #             print(words)
                    #             print(phrase)
                for phrase in gold_entities:
                    if phrase in pred_entities:
                        n_correct_absa += 1
                for phrase in gold_entities_no_senti:
                    if phrase in pred_entities_no_senti:
                        n_correct_ae += 1    
                n_preds += len(pred_entities)
                n_total += len(gold_entities)
        precision, recall, F1 = get_f1(n_correct_absa, n_preds, n_total)
        precision_ae, recall_ae, F1_ae = get_f1(n_correct_ae, n_preds, n_total)
        test_result = {
            'absa': {
                'precision': precision,
                'recall': recall,
                'F1': F1,
                'TP': n_correct_absa
            },
            'ae': {
                'precision': precision_ae,
                'recall': recall_ae,
                'F1': F1_ae,
                'TP': n_correct_ae
            },
            'number_preds': n_preds,
            'number_total': n_total
        }
        return test_result


# class BaseSentiTester(object):
#     def __init__(self, args, model, test_loader):
#         self.args = args
#         self.model = model
#         self.test_loader = test_loader
#         self.loss_func = torch.nn.CrossEntropyLoss()

#     def test_one_epoch(self, device, epoch):
#         # loss_meter = AverageMeter()
#         self.loss_func = self.loss_func.to(device)
#         self.model.eval()
#         n_correct_absa, n_correct_ae, n_preds, n_total = 0, 0, 0, 0
#         pred_list, label_list, pos_list = [], [], []
#         with torch.no_grad():
#             for j, inputs in enumerate(self.test_loader):
#                 inputs = {k: inputs[k].to(device) for k in inputs}
#                 logits, senti_logits = self.model(input_ids=inputs['input_ids'], masks=inputs['masks'])
#                 # loss = self.loss_func(logits.view(-1, self.args.n_labels), inputs['labels'].view(-1))
#                 # loss_meter.update(loss.item())
#                 l1, l2 = parse(logits, inputs['labels'], inputs['masks'])
#                 pos_list += inputs['pos_ids'].tolist()
#                 batch_size = inputs['input_ids'].shape[0]
#                 senti_preds = torch.argmax(senti_logits, dim=-1).tolist()
#                 for i in range(batch_size):
#                     if senti_preds[i]:
#                         # print(l1[i])
#                         l1[i] = [(b, e, senti_preds[i]) for b, e, t in l1[i]]
#                         # print(l1[i])
#                 pred_list += l1
#                 label_list += l2

#         raw_inputs = read_data(self.args.root, self.args.target, 'test')
#         n = 0
#         all_entity = set()
#         assert len(raw_inputs) == len(pred_list)
#         for (words, tags), pred, label, pos in zip(raw_inputs, pred_list, label_list, pos_list):
#             n_correct_absa += match_sentiment(pred, label)
#             n_correct_ae += match(pred, label)
#             n_preds += len(pred)
#             n_total += len(label)
#             pred_entities, gold_entities = [], []
#             for begin, end, senti in pred:
#                 raw_pos = set([pos[i+1] for i in range(begin, end+1)])
#                 pred_entity = ' '.join([words[i] for i in raw_pos])
#                 pred_entities.append((pred_entity, senti, min(raw_pos), max(raw_pos)))
#             for begin, end, senti in label:
#                 raw_pos = set([pos[i+1] for i in range(begin, end+1)])
#                 gold_entity = ' '.join([words[i] for i in raw_pos])
#                 gold_entities.append((gold_entity, senti, min(raw_pos), max(raw_pos)))
#             # print(' '.join(words))
#             # print(pred)
#             # print('pred entity: ' + ' | '.join([f'{e}({t})' for e, t, _, _ in pred_entities]))
#             # print('gold entity: ' + ' | '.join([f'{e}({t})' for e, t, _, _ in gold_entities]))
#             assert len(label) == len(gold_entities)
#             for item, phrase in zip(label, gold_entities):
#                 if phrase in pred_entities:
#                     # if item not in pred:
#                     #     print(' '.join(words))
#                     #     print(pred)
#                     #     print(phrase, item)
#                     n += 1
#             for item in gold_entities:
#                 all_entity.add(item[0])
#         precision, recall, F1 = get_f1(n, n_preds, n_total)
#         # print(F1)
#         # print(n, n_correct_absa, n_correct_ae, n_preds, n_total)
#         # precision, recall, F1 = get_f1(n_correct_absa, n_preds, n_total)
#         precision_ae, recall_ae, F1_ae = get_f1(n_correct_ae, n_preds, n_total)
#         return F1 * 100, precision * 100, recall * 100, F1_ae * 100, all_entity


tester_factory = {
    'baseline': BaseTester,
    # 'baseline_senti_cls': BaseSentiTester,
    'semi': BaseTester,
    'DA': BaseTester,
    'ner_dict': BaseTester,
    'lstm_DA': BaseTester
}