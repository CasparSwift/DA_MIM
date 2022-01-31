import re
import os

filenames = list(os.listdir('../BRIDGE-main/'))
filenames.sort()
for filename in filenames:
	d = {}
	if not filename.endswith('.out'):
		continue
	with open('../BRIDGE-main/' + filename, 'r') as f:
		print(filename)
		source, target ='', '' 
		for line in f:
			if 'processing train files: ' in line:
				source = re.findall(r'\./data/(.*?)/', line)[0]
			if 'processing test files' in line:
				target = re.findall(r'\./data/(.*?)/', line)[0]
			if line.startswith('aspect f1='):
				f1 = float(re.findall(r'aspect f1=(.*?),', line)[0])
				if source and target:
					# print(source, target, f1)
					if (source, target) in d:
						d[(source, target)].append(f1)
					else:
						d[(source, target)] = [f1]
					source = ''
					target = ''

	avgs = []
	for k in d:
		avg = sum(d[k])/len(d[k])
		avgs.append(avg)
		s, t = k
		print(s[0].upper() + ' to ' +t[0].upper(), avg, d[k])