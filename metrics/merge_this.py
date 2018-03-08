import sys
import glob
import rouge
import pandas as pd
import bleu

def preprocess(doc):	
	new_doc = []
	for i in doc:
		if isinstance(i,float):
			new_doc.append(" ")
		else:
			new_doc.append(i)

	return new_doc
			
fname1 = sys.argv[1] #decoded
data = pd.read_csv(fname1)
decoded = preprocess(data['prediction'])
ref = preprocess(data['ground_truths'])

bl = bleu.moses_multi_bleu(decoded,ref)
#
#bl = 0
#print(bl)
x = rouge.rouge(decoded,ref)
#print(x)
print('%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f'%(bl,x['rouge_1/f_score'],x['rouge_1/p_score'],x['rouge_1/r_score'],x['rouge_2/f_score'],x['rouge_2/p_score'],x['rouge_2/r_score'],x['rouge_l/f_score'],x['rouge_l/p_score'],x['rouge_l/r_score']))