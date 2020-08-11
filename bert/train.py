import pandas as pd 
import pdb

from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

import torch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from transformers import BertTokenizer, BertModel

# Parameters:
epochs = 100
batch_size = 32
lr = 0.01
train_size = 0.8

log_file = open('log.txt', 'w')
log_file.close()

# Loading Data
df = pd.read_csv('../qacc_data.csv')
questions = df['Question'].tolist()
acceptability = df['Q_Acc'].tolist()

train_data = questions[: int(len(questions) * train_size)]
train_labels = acceptability[: int(len(acceptability) * train_size)]

val_size = int(len(acceptability[int(len(acceptability) * train_size) : ]) / 2)

val_data = questions[int(len(questions) * train_size) : int(len(questions) * train_size) + val_size]
val_labels = acceptability[int(len(acceptability) * train_size) : int(len(questions) * train_size) + val_size]

test_data = questions[int(len(questions) * train_size) + val_size : ]
test_labels = acceptability[int(len(acceptability) * train_size) + val_size : ]

print('Train size', len(train_data))
print('Val size', len(val_data))
print('Test size', len(test_data))

# Model description
class linear(nn.Module):
	def __init__(self, cls_dim, hidden_dim, nClasses):
		super(linear, self).__init__()
		self.mlp = nn.Linear(cls_dim, nClasses)
	def forward(self, cls):
		out = self.mlp(cls)
		out = out.view(1,-1)
		return out


# Initializing model
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

bertmodel = BertModel.from_pretrained('bert-base-uncased')
bertmodel.eval()
bertmodel = bertmodel.to(device)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

model = linear(768, 254, 2)
model = model.to(device)


# Optimizer
# Defining loss function
criterion = nn.CrossEntropyLoss()
# Defining optimizer with all parameters
optimizer = torch.optim.Adam(model.parameters(), lr = lr)


def single_epoch(data, labels, train_flag):
	count = 0
	total_loss = 0
	predcited = []
	golden = []
	optimizer.zero_grad()

	for i in range(len(data)):
		count += 1

		# if count % 10 == 0:
		# 	print(total_loss/count)

		question = '[CLS]' + data[i] + '[SEP]'
		acc = int(labels[i])

		tokenized_text = tokenizer.tokenize(question)
		indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
		tokens_tensor = torch.tensor([indexed_tokens]).to(device)

		cls_state = bertmodel(tokens_tensor)[0][0][0]

		pred_classes = model(cls_state)

		true_class = torch.tensor([acc]).to(device)

		# pdb.set_trace()
		loss = criterion(pred_classes, true_class)
		loss.backward()
		total_loss += loss.item()

		golden.append(acc)
		predcited.append(torch.argmax(pred_classes).item())

		if (count % batch_size == 0) and train_flag:
			optimizer.step()
			optimizer.zero_grad()

	optimizer.step()
	optimizer.zero_grad()

	avg_loss = total_loss/count
	return avg_loss, golden, predcited


# pdb.set_trace()


prev_val_loss = 1000000000

for e in range(epochs):
	train_loss, train_golden, train_predicted = single_epoch(train_data, train_labels, train_flag = True)
	val_loss, val_golden, val_predicted = single_epoch(val_data, val_labels, train_flag = False)

	out_line = 'For epoch ' + str(e) + ' Train loss: ' + str(train_loss) + ' Train acc: ' + \
				str(accuracy_score(train_golden, train_predicted)) + ' | Val loss: ' + str(val_loss) + \
				' Val acc: ' + str(accuracy_score(val_golden, val_predicted)) + '\n\n'
	print(out_line)
	log_file = open('./log.txt', 'a')
	log_file.write(out_line)
	log_file.close()

	if val_loss < prev_val_loss:
		prev_val_loss = val_loss
		test_loss, test_golden, test_predicted = single_epoch(test_data, test_labels, train_flag = False)
		
		out_line = 'Test loss: ' + str(test_loss) + ' Test acc: ' + str(accuracy_score(test_golden, test_predicted)) + '\n'
		print(out_line)
		log_file = open('./log.txt', 'a')
		log_file.write(out_line)
		log_file.close()

		best_log = open('./best_log.txt', 'w')
		best_log.write('For Epoch: ' + str(e) + '\n\n')
		best_log.write(classification_report(train_golden, train_predicted))
		best_log.write('\n\n')
		best_log.write(classification_report(val_golden, val_predicted))
		best_log.write('\n\n')
		best_log.write(classification_report(test_golden, test_predicted))
		best_log.write('\n\n')
		best_log.close()

		torch.save(model.state_dict(), './trained_model/cls_best_linear_model.pt')


