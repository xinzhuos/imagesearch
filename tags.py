
import numpy as np

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

supercategory_dict = {item:val for val, item in enumerate(supercategory_set)}
category_dict = {item:val+1 for val, item in enumerate(category_set)}

train_tags = []

for i in range (10):
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

print train_tags
