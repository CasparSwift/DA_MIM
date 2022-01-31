import numpy as np
import json
import re


def parse(file):
	with open(file, 'r') as f:
		items = f.read().split('--------------------\n')
	return items

def parse_items(items):
	s = items[0]
	pred = items[1].replace('pred entity: ', '').split(' | ')
	gold = items[2].replace('gold entity: ', '').split(' | ')
	return s, pred, gold

def get_MI(probs1):
	x = np.array([np.mean(np.array(json.loads(p)), axis=0) for p in probs1])
	condi = np.mean(np.sum(-x * np.log2(x), axis=-1))
	x = np.mean(x, axis=0)
	y_entropy = sum([-i * np.log2(i) for i in x])
	mi = y_entropy - condi
	print(x, y_entropy, condi, mi)
	return mi

base_items = parse('base.out')
mim_items = parse('mim.out')
for ii1, ii2 in zip(base_items, mim_items):
	i1 = ii1.split('\n')
	i2 = ii2.split('\n')
	if len(i1) != 4:
		continue
	s1, pred1, gold1 = parse_items(i1)
	s2, pred2, gold2 = parse_items(i2)
	if pred1 != pred2 and pred2 != gold2 and len(gold2) > 1:
		words1 = re.findall(r'(.*?)\(\[.*?\]\)\s', s1)
		probs1 = re.findall(r'\((.*?)\)', s1)
		probs2 = re.findall(r'\((.*?)\)', s2)
		mi1 = get_MI(probs1)
		mi2 = get_MI(probs2)
		print(' '.join(words1))
		print(pred1)
		print(pred2)
		print(gold2)
		print('-'*10)

