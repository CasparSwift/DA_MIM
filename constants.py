# dataset settings
absa_domains = ['device', 'laptop', 'rest', 'service']
absa_label_map = {
	'O': 0, 
	'T-POS': 1, 
	'T-NEU': 2, 
	'T-NEG': 3
}
absa_bio_label_map = {
	'O': 0, 
	'B-POS': 1, 
	'B-NEU': 2, 
	'B-NEG': 3,
	'I-POS': 1, 
	'I-NEU': 2, 
	'I-NEG': 3,
	'O-POS': 1, 
	'O-NEU': 2, 
	'O-NEG': 3

}

ae_label_map = {
	'O': 0, 
	'T-POS': 1, 
	'T-NEU': 1, 
	'T-NEG': 1
}


ner_domains = ['CBS', 'CoNLL']
ner_label_map = {
	'O': 0,
	'PER': 1,
	'LOC': 2,
	'ORG': 3,
	'MISC': 4
}
pad_token = 0
cls_token = 101
sep_token = 102

m = {
	'S': 'service',
	'R': 'rest',
	'R1': 'rest1',
	'R2': 'rest2',
	'R3': 'rest3',
	'L': 'laptop',
	'L1': 'laptop1',
	'L2': 'laptop2',
	'L3': 'laptop3',
	'D': 'device',
	'D1': 'device1',
	'D2': 'device2',
	'D3': 'device3',
	'CoNLL': 'CoNLL',
	'CBS': 'CBS',
	'CBS_labeled': 'CBS_labeled',
	'SQuAD': 'SQuAD',
	'NQ': 'NaturalQuestions'
}