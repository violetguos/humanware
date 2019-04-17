import numpy as np
from utils.boxes import extract_labels_boxes
import pickle


def real_res(pred):
	original_metadata = '/rap/jvb-000-aa/COURS2019/etudiants/data/humanware/Humanware_v1_1553272293/valid/avenue_valid_metadata_split.pkl'
	with open(original_metadata, 'rb') as jar:
		instance_file_data = pickle.load(jar)
	correct = 0
	meta_orignal_list  =  []
	for i in range(1000):
		meta_orignal, _ = extract_labels_boxes(instance_file_data[i]['metadata'])
		digits_target = meta_orignal
		number_true = 0
		# convert digit
		for i in range(len(digits_target)):
			if digits_target[i] != -1:
				number_true += digits_target[i] * 10 ** ((len(digits_target) - 1) - i)
		meta_orignal_list.append(number_true)
		if pred[i] == number_true:
			correct += 1
	print("correct", correct)
             
	meta_orignal_list = np.asarray(meta_orignal_list)

	print("meta_orignal", meta_orignal_list[0:10])
	np.savetxt('meta_orignal.txt', meta_orignal_list)

if __name__ == '__main__':
	pred = np.loadtxt('/home/user50/humanware/results/b3phut1_eval_pred.txt')
	real_res(pred)