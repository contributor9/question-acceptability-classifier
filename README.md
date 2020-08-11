# question-acceptability-classifier

qacc_data.csv contains the labeled data compiled using questions generated from 6 datasets and 9 methods. Each classification method uses this for training and testing question classifier.

## LSTM
After running train.py in lstm/ folder, the trained model is saved and information for each epoch is logged in log files.

## GGNN
There are two variants, default is sliding window as discussed in the paper. It can be trained by running train.py within ggnn/ folder. Other than sliding window, dependency parse based graph is also experimented. This can be trained by running train_dep.py in ggnn/ folder. All the respective models and log files will be saved for both variants.

## BERT
Similarly, BERT classifier can be trained by running train.py in bert/ folder and the trained model and logs will be saved.
