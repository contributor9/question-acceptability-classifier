# question-acceptability-classifier

`qacc_data.csv` contains the labeled data compiled using questions generated from 6 datasets and 9 methods. Each classification method uses this for training and testing question classifier.

## LSTM
After running `train.py` in `lstm/` folder, the trained model is saved and information for each epoch is logged in log files.

## BERT
Similarly, BERT classifier can be trained by running `train.py` in `bert/` folder and the trained model and logs will be saved.

## GGNN
There are two variants, default is sliding window as discussed in the [paper](https://www.aclweb.org/anthology/2020.acl-main.31.pdf). It can be trained by running `train.py` within `ggnn/` folder. Other than sliding window, dependency parse based graph is also experimented. This can be trained by running `train_dep.py` in `ggnn/` folder. All the respective models and log files will be saved for both variants.


## Results

| Method        | Validation Accuracy     | Test Accuracy  |
| ------------- |:-------------:|:-----:|
| LSTM      | 79.76 | 78.27 |
| BERT      | 80.23      |   77.33 |
| GGNN (Sliding Window) | 81.00      |    79.42 |
| GGNN (Dpendency Parse) | 77.63      |    76.39 |


## LICENSE
MIT
