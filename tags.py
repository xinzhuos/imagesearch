
import numpy as np
import csv
from sklearn.ensemble import RandomForestRegressor

# Find set of supercategories, categories 
supercategory_set = set()
category_set = set()
for i in range (10000):
	file = open("tags_train/" + str(i) + ".txt", "r")
	lines = file.readlines() 
	for  line in lines:
		words = line.strip().split(':')
		supercategory_set.add(words[0])
		category_set.add(words[1])
	file.close()

# mapping from (super) category to index
supercategory_dict = {item:val for val, item in enumerate(supercategory_set)}
category_dict = {item:val+1 for val, item in enumerate(category_set)}

# Vectorize tags
train_tags = []
for i in range (10000):
	file = open("tags_train/" + str(i) + ".txt", "r")
	lines = file.readlines() 
	row = np.zeros(len(supercategory_set))
	for line in lines:
		words = line.strip().split(':')		
		supercategory_column = supercategory_dict.get(words[0])
		category_index = category_dict.get(words[1])
		row[supercategory_column] = category_index
	train_tags.append(row)
	file.close()

print ("Finished loading tags")

# Reading 1000-d train features 
features_train = np.zeros((10000,1000))
with open('features_train/features_resnet1000_train.csv', 'r') as csvfile:
	csv_reader = csv.reader(csvfile)
	for line in csv_reader:
		image_name = line[0].strip(".jpg")[13:]
		row = []
		for i in range(len(line)):
			if i > 0:
				row.append(float(line[i]))		
		features_train[int(image_name)] = row

query = []
with open('query_glove.csv', 'r') as csvfile:
	csv_reader = csv.reader(csvfile)
	for line in csv_reader:
		query.append(line)
print ("Finished loading queries")

clf = RandomForestRegressor()
clf.fit(query,train_tags)



