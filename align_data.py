from fuzzywuzzy import process 
import sys


def lower_case(line):
	sent = line.split('####')[0]
	words_and_labels = [item for item in line.split('####')[-1].split()]
	words, labels = [], []
	for items in words_and_labels:
		idx = items.rfind('=')
		words.append(items[:idx].lower())
		labels.append(items[idx+1:])
	out = sent.lower() + '####' + ' '.join(w.lower() + '=' + l for w, l in zip(words, labels))
	return out


for domain in ['rest', 'laptop', 'device']:
	for split_cnt in [1,2,3]:
		all_data_line = []
		with open(f'data/ABSA/{domain}_train.txt', 'r') as f:
			for line in f:
				sent = line.strip('\n')
				all_data_line.append(sent)
		with open(f'data/ABSA/{domain}_test.txt', 'r') as f:
			for line in f:
				sent = line.strip('\n')
				all_data_line.append(sent)

		all_data_line = list(map(lower_case, all_data_line))

		sent2id = {''.join(d.split('####')[0].split()): i for i, d in enumerate(all_data_line)}
		print(len(sent2id))

		has_matched_id = []

		unmatch_train = []
		new_train_data = []
		with open(f'data/ABSA_bridge/{domain}/train{split_cnt}/sentence.txt', 'r') as f:
			for i, line in enumerate(f):
				sent = line.strip('\n')
				key = ''.join(sent.split())
				if key in sent2id:
					new_train_data.append(all_data_line[sent2id[key]])
					has_matched_id.append(sent2id[key])
				else:
					new_train_data.append(sent)
					unmatch_train.append((i, key))

		unmatch_test = []
		new_test_data = []
		with open(f'data/ABSA_bridge/{domain}/test{split_cnt}/sentence.txt', 'r') as f:
			for i, line in enumerate(f):
				sent = line.strip('\n')
				key = ''.join(sent.split())
				if key in sent2id:
					new_test_data.append(all_data_line[sent2id[key]])
					has_matched_id.append(sent2id[key])
				else:
					new_test_data.append(sent)
					unmatch_test.append((i, key))

		has_matched_id = set(has_matched_id)

		unmatch_keys = [key for key, idx in sent2id.items() if idx not in has_matched_id]

		print(len(unmatch_keys))

		for idx, key in unmatch_train:
			best_match_key, score = process.extract(key, sent2id.keys(), limit=1)[0]
			sent = all_data_line[sent2id[best_match_key]] # may have typos, replace the sents
			new_line = new_train_data[idx] + '####'
			words = new_train_data[idx].split()
			labels = [item.split('=')[-1] for item in sent.split('####')[-1].split()]
			if len(words) == len(labels):
				new_line += ' '.join(w + '=' + l for w, l in zip(words, labels))
			else:
				if 'O' in set(labels) and len(set(labels)) == 1:
					new_line += ' '.join(w + '=O' for w in words)
				else:
					word_and_labels = [item.split('=') for item in sent.split('####')[-1].split()]
					word2label = {w: l for w, l in word_and_labels if l != 'O'}
					new_line += ' '.join(w + '=' + (word2label[w] if w in word2label else 'O') for w, l in zip(words, labels))
					print(new_line, file=sys.stderr)
			new_train_data[idx] = new_line
			if score < 90:
				print(score)
				print(key)
				print(best_match_key)
				print(new_line)
				print('-'* 50)


		for idx, key in unmatch_test:
			best_match_key, score = process.extract(key, sent2id.keys(), limit=1)[0]
			if score < 90:
				print(score)
				print(key)
				print(best_match_key)
				print('-'* 50)
			sent = all_data_line[sent2id[best_match_key]] # may have typos, replace the sents
			new_line = new_test_data[idx] + '####'
			words = new_test_data[idx].split()
			labels = [item.split('=')[-1] for item in sent.split('####')[-1].split()]
			if len(words) == len(labels):
				new_line += ' '.join(w + '=' + l for w, l in zip(words, labels))
			else:
				if 'O' in set(labels) and len(set(labels)) == 1:
					new_line += ' '.join(w + '=O' for w in words)
				else:
					word_and_labels = [item.split('=') for item in sent.split('####')[-1].split()]
					word2label = {w: l for w, l in word_and_labels if l != 'O'}
					new_line += ' '.join(w + '=' + (word2label[w] if w in word2label else 'O') for w, l in zip(words, labels))
					print(new_line, file=sys.stderr)
			new_test_data[idx] = new_line

		with open(f'data/ABSA/{domain}{split_cnt}_train.txt', 'w') as f:
			f.write('\n'.join(new_train_data))
		with open(f'data/ABSA/{domain}{split_cnt}_test.txt', 'w') as f:
			f.write('\n'.join(new_test_data))
			


