import pickle
import gzip
import json
import pandas as pd
import numpy as np
import sys
'''
mapper = pickle.load(open('ids_to_idxs.pkl','rb'))
'''

data_orignal = sys.argv[1] #test file
data_gen = sys.argv[2]#predicted file
data_json = sys.argv[3]
data_orignal = json.load(open(data_orignal))
data_gen = pickle.load(gzip.open(data_gen))
data_json = json.load(open(data_json))

idxs = data_json['idxs']
ids = data_json['ids']
mapper = {}

for k,v in zip (idxs,ids):
	mapper[k] = v

#write code for mapper as well
def get_data_id(data,key):
	found = None
	for i in range(0,len(data['data'])):

		if key == data['data'][i]['paragraphs'][0]['qas'][0]['id']:
			found  = i

	passage = data['data'][found]['paragraphs'][0]['context']
	true_span = data['data'][found]['paragraphs'][0]['qas'][0]['answers'][0]['text']
	if found is None:
		

		return None, None

	return passage,true_span


all_starts = data_gen['yp']
all_ends = data_gen['yp2']
all_ids = data_gen['idxs']
top_k = 5
all_spans = []
for i in range(top_k):
	all_spans.append([])
true_spans = []
example_ids = []
in_count = 0
out_count = 0

for id_, starts, ends in zip(all_ids,all_starts,all_ends):
	
	key = mapper[id_]
	passage,true_span = get_data_id(data_orignal,key)
	sorted_starts = np.argsort(starts[0])[::-1][:top_k]
	sorted_ends = np.argsort(ends[0])[::-1][:top_k]
	true_spans.append(true_span)
	example_ids.append(key)
	in_count = in_count +1 
	for j in range(0,top_k):
		out_count = out_count + 1
		
		if sorted_starts[j] >= sorted_ends[j]:
			all_spans[j].append(" ")
		else:
			all_spans[j].append(" ".join(passage.split()[sorted_starts[j]:sorted_ends[j]]))

top_k_spans_dict = {}
top_k_spans_dict['true_span'] = true_spans
top_k_spans_dict['example_id'] = example_ids



for k in range(0,top_k):
	top_k_spans_dict['pred_'+str(k+1)]  = all_spans[k]	


df = pd.DataFrame(top_k_spans_dict)

df.to_csv('span_wala_result.csv',index = False)







'''
import string
import re
from collections import Counter
x = 'i loved his madness and his dark comical role he doesnt even try but you cant help but laugh at a lot of his lines the way he looks and the way he presents every scene he didnt have a lot of movement he is confined to a wheel chair but he is so effective and perfect no one could have replaced him as lb hes a terrific actor grace kelly what a beauty beauty and talent what a great combination and she had it playing liza'
y = 'when lb talks of the murder to liza she is doubtful but never dismisses that it could be a possibility and stays with him into the end'

def normalize_answer(s):
	"""Lower text and remove punctuation, articles and extra whitespace."""
	def remove_articles(text):
		return re.sub(r'\b(a|an|the)\b', ' ', text)

	def white_space_fix(text):
		return ' '.join(text.split())

	def remove_punc(text):
		exclude = set(string.punctuation)
		return ''.join(ch for ch in text if ch not in exclude)

	def lower(text):
		return text.lower()

	return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score(prediction, ground_truth):
	prediction_tokens = normalize_answer(prediction).split()
	ground_truth_tokens = normalize_answer(ground_truth).split()
	common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
	num_same = sum(common.values())
	if num_same == 0:
		return 0
	precision = 1.0 * num_same / len(prediction_tokens)
	recall = 1.0 * num_same / len(ground_truth_tokens)
	f1 = (2 * precision * recall) / (precision + recall)
	return f1

print(f1_score(x,y))
'''