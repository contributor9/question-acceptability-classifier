import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import pdb


class vanila_lstm_classifier(nn.Module):
	def __init__(self, embed_size, nHidden, nClasses):
		super(vanila_lstm_classifier, self).__init__()
		self.embed_size = embed_size
		self.nHidden = nHidden

		self.lstm = nn.LSTM(embed_size, nHidden, bidirectional = True)
		self.out_linear = nn.Linear(2 * nHidden, nClasses)


	def forward(self, in_seq):
		in_seq = in_seq.view(-1, 1, self.embed_size)
		recurrent, (hidden, c) = self.lstm(in_seq)
		hidden = hidden.view(-1, 2*self.nHidden)

		out = self.out_linear(hidden)
		out = out.view(1,-1)

		return out