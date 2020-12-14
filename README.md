### ML Project2 Twitter Text Classification

#### Enviroment

The whole experiment is run under python 3.7.3 with
- jupyterlab 2.2.6
- numpy 1.19.1
- sklearn 0.23.2
- pytorch 1.7.0
- fasttext (```$ pip install fasttext```)
- transformers (```$ pip install transformers```)
- tqdm (```$ pip install tqdm```)
- (optional) GPU support for pytorch (we use GeForce GTX 1080 GPUs for our DL model training)

#### File Description

1. ```run.py``` : Run deep learning models and get kaggle submission. To switch models, you can edit this file as follow:
```python
# train XLnet and get the results
#flat_predictions = train_xlnet(x_train=x_train, y_train=y_train, batch_size=32, lr=2e-5, epochs=3, ids=ids, x_test=x_test)

# train Bert and get the results
flat_predictions = train_bert(x_train=x_train, y_train=y_train, batch_size=32, lr=2e-5, epochs=4, ids=ids, x_test=x_test)

```
2. ```run_ml.py``` : Run simple machine learning methods and get kaggle submission. To switch methods, you can edit this file as follow:
```python
# choose one simple ml classifier and train model
# gaussian naive bayes
classifier = GaussianNB()
# random forest
#classifier = RandomForestClassifier(n_estimators=300)
# SVM
#classifier = make_pipeline(StandardScaler(), SVC(gamma='auto'))

```
3. ```data_preprocess.py``` : Contains functions for data preprocessing
4. ```helpers.py``` : Useful functions to load data, compute word embedding and calculate time, etc.
5. ```bert.py``` : Function to train BERT base-uncased model and get the prediction 
6. ```xlnet.py``` : Function to train XLNet base-cased model and get the prediction
7. ```ml_cross_validation.jupyter``` : Do cross validation on traditional ML methods

#### Execute directly

To execute the program and reproduce our results, you can directly run
```bash
python3 run.py
```
in bash. The program will train BERT model as default and output the __submission.csv__ file in the output path. This file is our best prediction which can get 0.908 precision score on AICrowd. Note that if you want to use different training data, testing data, or output path, be sure to modify the variables in run_dl.py.

