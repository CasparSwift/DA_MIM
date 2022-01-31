import os
import json
import re


def get_f1(n_hit, n_preds, n_total):
    precision = n_hit / (n_preds + 1e-6)
    recall = n_hit / (n_total + 1e-6)
    F1 = 2 * precision * recall / (precision + recall + 1e-6)
    return precision * 100, recall * 100, F1 * 100


def get_results(method='MIM', task='AE'):
	absa_result, ae_result = dict(), dict()
	macro_result = dict()
	domain_pair = []
	for s in ['L', 'D', 'R']:
		for t in ['L', 'D', 'R']:
			if s == t:
				continue
			else:
				domain_pair.append([s, t])

	for s, t in domain_pair:
		f1s, f1_aes = [], []
		TP, TP_ae, Pred, Total = 0, 0, 0, 0

		overall_f1_aes = []
		
		if task == 'ABSA':
			names = [s+t]
		else:
			names = [f'{s}{i}{t}{i}' for i in [1, 2, 3]]
		for name in names:
			if method == 'MIM' and 'baseline' in name:
				continue
			if method == 'baseline' and 'baseline' not in name:
				continue

			f1_aes_cur_run = []
			if task =='ABSA':
				loggings = ['1113_ABSA', '1113_ABSA_2']
			else:
				loggings = ['1109_AE', '1111_AE', '1113_grid', '1113_grid_0.5', '1113_grid_2', '1113_grid_3', '1113_grid_4']
			loggings = ['BERT_E']
			for logging in loggings:
				path = os.path.join('output', name, logging)
				if not os.path.exists(path):
					continue

				aes = []
				for run in os.listdir(path):
					if not run.endswith('.txt'):
						continue
					with open(os.path.join(path, run), 'r') as f:
						test_result = json.loads(f.read())
						f1s.append(test_result['absa']['F1'])
						f1_aes.append(test_result['ae']['F1'])
						TP += test_result['absa']['TP']
						TP_ae += test_result['ae']['TP']
						Pred += test_result['number_preds']
						Total += test_result['number_total']
						aes.append(test_result['ae']['F1'])
				f1_aes_cur_run.append(aes)
				print(path, aes)

			# grid_search
			# try:
			# 	with open(f'grid_{name.lower()}.out', 'r') as f:
			# 		for line in f:
			# 			if line.startswith('Results:'):
			# 				f1_aes_grid = re.findall(r'f1_aes=(\[.*?\]),', line)[0]
			# 				f1_aes_grid = json.loads(f1_aes_grid)
			# 				f1_aes_cur_run.append(f1_aes_grid)
			# except Exception as e:
			# 	pass

			if not f1_aes_cur_run:
				continue

			if task == 'AE':
				f1_aes_cur_run = sorted(f1_aes_cur_run, key=lambda x: sum(x), reverse=True)[0]
				print(name, f1_aes_cur_run)
				overall_f1_aes += f1_aes_cur_run

		if not f1s:
			continue
		macro_f1 = sum(f1s)/len(f1s)
		macro_f1_ae = sum(f1_aes)/len(f1_aes)
		micro_f1 = get_f1(TP, Pred, Total)[-1]
		micro_f1_ae = get_f1(TP_ae, Pred, Total)[-1]
		# print(f'f1: {f1s} macro: {macro_f1:.2f} micro: {micro_f1:.2f}')
		# print(f'f1_ae: {f1_aes} macro: {macro_f1_ae:.2f} micro: {micro_f1_ae:.2f}')
		absa_result[s+t] = absa_result.get(s+t, []) + [(f1s, '%.2f' % (micro_f1))]
		ae_result[s+t] = ae_result.get(s+t, []) + [(f1_aes, '%.2f' % (micro_f1_ae))]
		macro_result[s+t] = (overall_f1_aes, '%.2f' % (sum(overall_f1_aes)/(len(overall_f1_aes)+1e-6)))
	for k in absa_result:
		absa_result[k] = sorted(absa_result[k], key=lambda x: float(x[-1]), reverse=True)[0]
	for k in ae_result:
		ae_result[k] = sorted(ae_result[k], key=lambda x: float(x[-1]), reverse=True)[0]
	return absa_result, ae_result, macro_result


if __name__ == '__main__':
	r1, r2, r3 = get_results(method='MIM', task='AE')
	for k, v in r3.items():
		print(k, v)

