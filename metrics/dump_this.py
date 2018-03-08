import glob
from evaluate_and_dump import evaluate_and_dump
import pandas as pd
import numpy as np
import sys
truth = sys.argv[1]
pred = sys.argv[2]
e, f = evaluate(truth,pred)
print(f)
print(e)
#d = {'f1':f1,'exact match':em, 'steps':steps}

#df = pd.DataFrame(d)
#df.to_csv('squad_f1.csv',index=False)