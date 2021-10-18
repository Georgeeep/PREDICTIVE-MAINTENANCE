# PREDICTIVE-MAINTENANCE
APPLICATIONS OF DEEP LEARNING IN PREDICTIVE MAINTENANCE SCENARIOS
There are four python files each pertaining to a different model:
  ann.py- Tensorflow Feed Forward Neural Network
  gru.py- Tensorflow GRU RNN Neural Network
  rnn.py- Tensorflow LSTM RNN Neural Network
  tslearnsvr.py- Sklearn linear and polynomial regressions and tslearn SVR
 Each python file is self contained looping through all datasets and should run on start provided all required files exist within the same directory
 The required files are:
    The ones that start with 'RUL'- These are the truth sets that the test set requires
    The ones that start with 'train_'- the train sets per dataset
    The ones that start with 'test_' - the test sets per dataset
    final_test.txt- the test set whose truth set was not released which the competition was based on
    
There are 3 types of folders logs, figures and models. They contain the: Tensorboard logs, the figures produced and the Modelcheckpoints for the produced figures per Tensorflow Model discussed.

result text files pretty print the metrics per timestep/batchsize combination per model per dataset

WEKAresults contains all experiments run on tensorflow as well as their results
