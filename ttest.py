from scipy import stats

a = [47.92,49.79 ,47.64 ,35.14 ,38.14 ,37.22 , 38.68 , 33.69 , 27.46 , 34.08]
b = [50.20, 53.24, 54.98, 42.78, 43.20, 46.69, 38.20, 32.49, 35.87, 35.38]

r = stats.ttest_rel(a, b)
print(r)