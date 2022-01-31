from show import get_results
from scipy.stats import ttest_rel

settings = 'S→R L→R D→R R→S L→S D→S R→L S→L R→D S→D'.replace('→', '').split()

class Line:
	def __init__(self, method_name, settings, performance):
		self.method_name = method_name
		self.settings = settings
		if isinstance(performance, list):
			self.performances = {s: ('base', [float(p)]*5, p) for s, p in zip(settings, performance)}
		elif isinstance(performance, dict):
			self.performances = {s: performance.get(s, '-') for s in settings}

	def to_str(self, s):
		return '%.2f' % s

	def get_avg(self):
		p = [float(self.performances[s].replace('\\textbf{', '').replace('}', '')) 
			for s in self.settings]
		return self.to_str(sum(p)/len(p))

	def to_latex(self, avg=True):
		items = [self.method_name] + [self.performances[s] for s in self.settings]
		if avg:
			items += [self.get_avg()]
		return ' & '.join(items) + ' \\\\'


class Block:
	def __init__(self, settings, lines):
		self.lines = lines
		self.settings = settings

	def to_latex(self, avg=True):
		for s in settings:
			ps = [line.performances[s] for line in self.lines]
			my_performance = ps[-1]

			sorted_ps = sorted(enumerate(ps), key=lambda x: float(x[1][-1]))
			max_pos, max_performance = sorted_ps[-1]
			second_pos, second_performance = sorted_ps[-2]

			# textbf
			old_tuple = self.lines[max_pos].performances[s]
			bf_str = f'\\textbf{{{old_tuple[2]}}}'
			new_tuple = (old_tuple[0], old_tuple[1], bf_str)
			self.lines[max_pos].performances[s] = new_tuple

			# significant testing
			# if max_performance[0] != 'base':
			print(s)
			# print(max_performance)
			print(second_performance)
			print(my_performance)
			p_value = ttest_rel(second_performance[1], max_performance[1])[1]
			print(p_value)
			print('-'*40)

		return '\n'.join([line.to_latex() for line in self.lines])


class Table_Head:
	def __init__(self, settings):
		self.settings = settings

	def to_latex(self, avg=True):
		if avg:
			return ' & '.join(['\\bf ' + s[0] + '$\\to$' + s[1] for s in self.settings] + ['\\bf AVG'])
		else:
			return ' & '.join(['\\bf ' + s[0] + '$\\to$' + s[1] for s in self.settings])


s = '''Hier-Joint
31.10
33.54
32.87
15.56
13.90
19.04
20.72
22.65
24.53
23.24
23.72
RNSCN
33.21
35.65
34.60
20.04
16.59
20.03
26.63
18.87
33.26
22.00
26.09
AD-SAL
41.03
43.04
41.01
28.01
27.20
26.62
34.13
27.04
35.44
33.56
33.71
BERT-base
44.66
40.38
40.32
19.48
25.78
30.31
31.44
30.47
27.55
33.96
32.44
BERT-DANN
45.84
41.73
34.68
21.60
25.10
18.62
30.41
31.92
34.41
23.97
30.79
BERT-UDA
47.09
45.46
42.68
33.12
27.89
28.03
33.68
34.77
34.93
32.10
35.98'''.split('\n')

s_ae = '''Hier-Joint
46.39
48.61
42.96
27.18
25.22
29.28
34.11
33.02
34.81
35.00
35.66
RNSCN
48.89
52.19
50.39
30.41
31.21
35.50
47.23
34.03
46.16
32.41
40.84
AD-SAL
52.05
56.12
51.55
39.02
38.26
36.11
45.01
35.99
43.76
41.21
43.91
BERT-base
54.29
46.74
44.63
22.31
30.66
33.33
37.02
36.88
32.03
38.06
37.60
BERT-DANN
54.32
48.34
44.63
25.45
29.83
26.53
36.79
39.89
33.88
38.06
37.77
BERT-UDA
56.08
51.91
50.54
34.62
32.49
34.52
46.87
43.98
40.34
38.36
42.97
'''.split('\n')

base_absa_result, base_ae_result = get_results('baseline')
our_absa_result, our_ae_result = get_results('MIM')


def get_latex(s, base_result, our_result, mode):
	table_head = Table_Head(settings)
	baseline_lines = [Line(s[i], settings, s[i+1:i+11]) for i in range(0, len(s), 12)]
	base_result_lines = Line('BERT-base$^*$', settings, base_result)
	our_result_lines = Line('BERT-MIM (ours)', settings, our_result)
	block = Block(settings, baseline_lines[:4] + [base_result_lines, baseline_lines[4], baseline_lines[5], our_result_lines])
	latex = f'''\\begin{{table*}}[t!]
\\centering
\\resizebox{{\\textwidth}}{{22mm}}{{
\\begin{{tabular}}{{{('l|' + 'c|'*11).strip('|')}}}
\\toprule[1.5pt]
\\bf Methods & {table_head.to_latex()} \\\\
\\midrule[1pt]
{block.to_latex()}
\\bottomrule[1.5pt]
\\end{{tabular}}
}}
\\caption{{{mode} results}}
\\label{{tab:{mode} results}}
\\end{{table*}}'''
	print(latex)


# get_latex(s, base_absa_result, our_absa_result, 'absa')
get_latex(s_ae, base_ae_result, our_ae_result, 'ae')

