import glob
from evaluate_off import evaluate
from evaluate_and_dump import evaluate_and_dump
import pandas as pd
import numpy as np
f1 = []
steps = np.arange(0,50) * 500
em = []
max_f1 = 0
file_for_computation = None
for i in sorted(glob.glob('dev-0*.json')):
	e,f = evaluate('dev-v1.1.json',i)
	if f > max_f1:
		max_f1 = f
		file_for_computation = i 

	f1.append(f)
	em.append(e)

d = {'f1':f1,'exact match':em, 'steps':steps}
df = pd.DataFrame(d)
df.to_csv('squad_f1.csv',index=False)
print(file_for_computation)

e, f = evaluate_and_dump('dev-v1.1.json',i) #file with max F1
print('exact match')
print(e)
print('exact match')
print(f)