import os
import json
from collections import defaultdict
from collections import Counter
import csv
import nltk
import numpy as np
import argparse

# Command line interface that accepts a model type and file to write errors to
def clip():
	parser = argparse.ArgumentParser()
	parser.add_argument('-m', '--model', type = str, help = 'model type')
	parser.add_argument('-e', '--error_file', type = str, help = 'path to save examples for error analysis')
	args = parser.parse_args()
	return args.model, args.error_file

# Writes dev set prediction errors to file path for further analysis
def write_errors(path, preds, actuals, index_map_dev):
	missed_dev_indices = []
	missed_preds = []
	for i, pred in enumerate(preds):
	    if pred != actuals[i]:
	        missed_dev_indices.append(i)
	        missed_preds.append(pred)

	responses = np.load('responses.npy')
	num_examples_to_analyze = 250
	n_missed = len(missed_preds)
	actual_n = min(num_examples_to_analyze, n_missed)
	indices_to_analyze = np.random.choice(range(n_missed), actual_n, replace = False)
	with open(path, 'w') as file:
		for i, ind in enumerate(indices_to_analyze):
		    missed_pred = missed_preds[ind]
		    missed_og_index = index_map_dev[ind]
		    file.write('\nMissed Example #' + str(i+1) + ' of ' + str(actual_n) + '\n')
		    if missed_pred == 0: #originally predicted the first comment to be non-sarcastic
		        file.write('Actual non-sarcastic comment: ' + str(responses[missed_og_index][1])+ '\n')
		        file.write('Word count: ' + str(len((responses[missed_og_index][1]).split()))+ '\n')
		        file.write('Actual sarcastic comment: ' + str(responses[missed_og_index][0])+ '\n')
		        file.write('Word count: ' + str(len((responses[missed_og_index][0]).split()))+ '\n')

		    else:
		        file.write('Actual non-sarcastic comment: ' + str(responses[missed_og_index][0])+ '\n')
		        file.write('Word count: ' + str(len((responses[missed_og_index][0]).split()))+ '\n')
		        file.write('Actual sarcastic comment: ' + str(responses[missed_og_index][1])+ '\n')
		        file.write('Word count: ' + str(len((responses[missed_og_index][1]).split()))+ '\n')

# Selects indices from the original dataset for training, builds a map to go from indices in the training set
# to indices in the original dataset, which is used for error analysis
def get_training_indices(data, n, frac = 0.95, seed = False):
	if seed:
		np.random.seed(224)
	train_indices = np.random.choice(range(n), int(round(frac*n)), replace = False)
	train_indices_subset = np.random.choice(train_indices, int(round(frac*n)), replace = False)
	elmo_train = []
	for i in train_indices:
	    if i in train_indices_subset:
	        elmo_train.append(data[i])
	print('length of training list: ', len(elmo_train))
	index_map_train = {}
	for i in range(len(train_indices_subset)):
	    index_map_train[i] = train_indices_subset[i]
	print('size of training set:', str(len(index_map_train)))
	return elmo_train, index_map_train, train_indices

# Selects indices from the original dataset for development, builds a map to go from indices in the dev set
# to indices in the original dataset, which is used for error analysis
def get_dev_indices(data, train_indices, n, frac = 0.5):
	other_indices = list(set(range(n)) - set(train_indices))
	dev_indices = np.random.choice(other_indices, int(round(0.5*len(other_indices))), replace = False)
	elmo_dev = []
	for i in other_indices:
	    if i in dev_indices:
	        elmo_dev.append(data[i])
	print('length of dev list: ', len(elmo_dev))
	index_map_dev = {}
	for i in range(len(dev_indices)):
	    index_map_dev[i] = dev_indices[i]
	print('size of dev set:', str(len(index_map_dev)))
	return elmo_dev, index_map_dev, dev_indices, other_indices

# Selects indices from the original dataset for testing, builds a map to go from indices in the test set
# to indices in the original dataset, which is used for error analysis
def get_test_indices(data, other_indices, dev_indices):
	test_indices = list(set(other_indices) - set(dev_indices))
	elmo_test = [data[i] for i in test_indices]
	print('length of test list: ', len(elmo_test))
	index_map_test = {}
	for i in range(len(test_indices)):
	    index_map_test[i] = test_indices[i]
	print('size of test set:', str(len(index_map_test)))
	return elmo_test, index_map_test

# processes the raw text into pairs of sentences, which are used as inputs for generating the word embeddings
# and pulling example comments for error analysis
def get_textual_examples(data_dir):
	# Read in Data
	comments_file = os.path.join(data_dir, 'comments.json')
	train_file = os.path.join(data_dir, 'train-balanced.csv')

	with open(comments_file, 'r') as f:
		comments = json.load(f)

	# Format data
	train_ancestors = []
	train_responses = []
	train_labels = []
	with open(train_file, 'r') as f:
		reader = csv.reader(f, delimiter='|')
		for row in reader:
			ancestors = row[0].split(' ')
			responses = row[1].split(' ')
			labels = row[2].split(' ')
			train_ancestors.append([comments[r]['text'].lower() for r in ancestors])
			train_responses.append([comments[r]['text'].lower() for r in responses])
			train_labels.append(labels)

	train_vocab = defaultdict(int)
	for pair in train_responses:
		for comment in pair:
			for w in nltk.word_tokenize(comment):
				train_vocab[w] += 1
	train_vocab = Counter(train_vocab)
	responses = train_responses
	np.save('responses.npy', responses)
	print('saved responses')
	return responses