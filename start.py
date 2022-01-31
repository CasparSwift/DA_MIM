import os
import time
import argparse
import json
from datetime import datetime
from constants import m

parser = argparse.ArgumentParser("Domain Adaptation")
parser.add_argument("--runs", type=int, default=1)
parser.add_argument("--s", type=str, default='L')
parser.add_argument("--t", type=str, default='R')
parser.add_argument("--logging", type=str)
parser.add_argument("--debug", action="store_true")
args = parser.parse_args()


with open('bert_base_fmim_config.json', 'r') as f:
	config = json.loads(f.read())


def run(s, t, lamb, thres, wd, model_type='DA', runs=0, cuda=0, backbone='b', lr=2e-5):
	output_dir = os.path.join('output', f'{s}{t}', args.logging)
	os.makedirs(output_dir, exist_ok=True)
		
	name = f'run[{runs}]'
	if s == ['CoNLL', 'CoNLL_labeled']:
		dataset, n_labels, epochs_num, bsz = 'NER', 5, 3, 12
	elif s in ['SQuAD', 'NQ']:
		dataset, n_labels, epochs_num, bsz = 'QA', 3, 5, 64
	elif s in ["L", "D", "R", "S"]:
		dataset, n_labels, epochs_num, bsz = 'ABSA', 4, 20, 8
	else:
		if model_type == 'lstm_DA':
			dataset, n_labels, epochs_num, bsz = 'AE', 2, 30, 8
		else:
			dataset, n_labels, epochs_num, bsz = 'AE', 2, 20, 8

	if backbone == 'b':
		model_dir = 'bert-base-uncased'
	else:
		model_dir = 'rest_pt_review'

	if model_type != 'lstm_DA':
		cmd = f'''
			CUDA_VISIBLE_DEVICES={cuda} python3 main.py --model_type={model_type} --dataset={dataset} --n_labels={n_labels} \
				--source={m[s]} --target={m[t]} --output_dir=\"{output_dir}\" --output_name={name} \
				--batch_size={bsz} --test_batch_size={bsz} --logging_step=50 --epoch_num={epochs_num} \
				--model_dir=../{model_dir}/ --weight_decay={wd} --balance=1 --learning_rate={lr} \
				--lambda_MI={lamb} --MI_threshold={thres} --random_seed={runs+1024}
		'''.replace('\n', '').replace('\t', '')
	else:
		cmd = f'''
			CUDA_VISIBLE_DEVICES={cuda} python3 main.py --model_type={model_type} --dataset={dataset} --n_labels={n_labels} \
				--source={m[s]} --target={m[t]} --output_dir=\"{output_dir}\" --output_name={name} \
				--batch_size={bsz} --test_batch_size={bsz} --logging_step=50 --epoch_num={epochs_num} \
				--lambda_MI={lamb} --MI_threshold={thres} --random_seed={runs+1024} \
				--learning_rate=1e-4 --weight_decay=1e-5 --max_sent_length=128
		'''.replace('\n', '').replace('\t', '')
	if args.debug:
		cmd += ' --debug'
	print(cmd)
	os.system(cmd)

	print('------')
	f1s = []
	f1_aes = []
	for i in range(3):
		try:
			with open(os.path.join(output_dir, f'run[{i}]_results.txt'), 'r') as f:
				test_result = json.loads(f.read())
				f1s.append(test_result['absa']['F1'])
				f1_aes.append(test_result['ae']['F1'])
		except:
			pass

	print(f'f1: {f1s} avg: {sum(f1s)/3:.2f}')
	print(f'f1_ae: {f1_aes} avg: {sum(f1_aes)/3:.2f}')
	print(f'Results: lamb={lamb}, thres={thres}, wd={wd}, f1_aes={f1_aes}, avg={sum(f1_aes)/3}')
	return f1_aes


def main():
	for i in range(1):
		run('R1', 'L1', lamb=0.015, thres=0.7, wd=0.1, model_type='DA', runs=i, cuda=4, lr=2e-5)
	
	# for s in [1, 2, 3]:
	# 	for i in range(3):
	# 		c = config["DR"]
	# 		lamb, thres, wd = c["lamb"], c["thres"], c["wd"]
	# 		run(f'D{s}', f'R{s}', lamb=lamb, thres=thres, wd=wd, model_type='DA', runs=i, cuda=1, backbone='e', lr=2e-5)
	
	# run('CoNLL', 'CBS_labeled', lamb=0, thres=0.5, model_type='ner_dict')
	# for lamb in [1e-3, 1e-4, 1e-5, 1e-6]:
	# run('SQuAD', 'NQ', lamb=0, thres=0, model_type='baseline')

	# for s in ['S', 'L', 'D', 'R']:
	# 	for t in ['R']:
	# 		if s == t:
	# 			continue
	# for i in range(5):
	# 	run('S', 'R', lamb=0.02, thres=0.5, wd=0.01, model_type='DA', runs=i, cuda=0)
	

if __name__ == '__main__':
	main()

		

