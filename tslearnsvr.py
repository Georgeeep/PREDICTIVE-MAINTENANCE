from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import pandas as pd
import copy
import numpy as np

from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error
from matplotlib import pyplot
from sklearn.preprocessing import LabelEncoder
import os
import pprint
from sklearn import preprocessing
from tslearn import svm as svm

datalists = ['FD001', 'FD002', 'FD003', 'FD004']
label_encoder = LabelEncoder()
outerdict = {}
epochs = [200]
############################import datasets################################################
for datalist in datalists:
    train_dataog = pd.read_csv('train_' + datalist + '.txt', delimiter=' ', header=None)
    test_dataog = pd.read_csv('test_' + datalist + '.txt', delimiter=' ', header=None)
    finaltestog = pd.read_csv('final_test.txt', delimiter=' ', header=None)
    ###################drop last two lines######################################
    finaltestog.drop(finaltestog.columns[[26, 27]], axis=1, inplace=True)
    train_dataog.drop(train_dataog.columns[[26, 27]], axis=1, inplace=True)
    test_dataog.drop(test_dataog.columns[[26, 27]], axis=1, inplace=True)
    truth = pd.read_csv('RUL_' + datalist + '.txt', sep=" ", header=None)
    truth.drop(truth.columns[[1]], axis=1, inplace=True)
    ###################################per-loop initializations################################
    innerdict = {}
    resultsdict = dict()
    bestlistmse = []
    optimaltimestep = 100000
    bestlistmae = []
    bestlistmape = []
    rmsepereng = {}
    rmseperengskipped = {}
    bestrmsepereng = {}
    #######################name the columns#######################################

    train_dataog.columns = ['id', 'cycle', 'setting1', 'setting2', 'setting3', 's1', 's2', 's3',
                            's4', 's5', 's6', 's7', 's8', 's9', 's10', 's11', 's12', 's13', 's14',
                            's15', 's16', 's17', 's18', 's19', 's20', 's21']
    finaltestog.columns = ['id', 'cycle', 'setting1', 'setting2', 'setting3', 's1', 's2', 's3',
                           's4', 's5', 's6', 's7', 's8', 's9', 's10', 's11', 's12', 's13', 's14',
                           's15', 's16', 's17', 's18', 's19', 's20', 's21']
    test_dataog.columns = ['id', 'cycle', 'setting1', 'setting2', 'setting3', 's1', 's2', 's3',
                           's4', 's5', 's6', 's7', 's8', 's9', 's10', 's11', 's12', 's13', 's14',
                           's15', 's16', 's17', 's18', 's19', 's20', 's21']
    colums = ['setting1', 'setting2', 'setting3', 's1', 's2', 's3',
              's4', 's5', 's6', 's7', 's8', 's9', 's10', 's11', 's12', 's13', 's14',
              's15', 's16', 's17', 's18', 's19', 's20', 's21',
              'cycle_norm']

    ########################RUL creation train#################################
    rul = pd.DataFrame(train_dataog.groupby('id')['cycle'].max()).reset_index()
    rul.columns = ['id', 'max']
    train_dataog = train_dataog.merge(rul, on=['id'], how='left')
    train_dataog['RUL'] = train_dataog['max'] - train_dataog['cycle']
    train_dataog.drop('max', axis=1, inplace=True)
    ########################train normalization################################

    train_dataog['cycle_norm'] = train_dataog['cycle']
    scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1))  # works better for tanh
    train_dataog[colums] = scaler.fit_transform(train_dataog[colums])

    ########################RUL creation finaltest#################################

    rul = pd.DataFrame(finaltestog.groupby('id')['cycle'].max()).reset_index()
    rul.columns = ['id', 'max']
    finaltestog = finaltestog.merge(rul, on=['id'], how='left')
    finaltestog['RUL'] = finaltestog['max'] - finaltestog['cycle']
    finaltestog.drop('max', axis=1, inplace=True)
    ########################finaltest normalization################################

    finaltestog['cycle_norm'] = finaltestog['cycle']

    finaltestog[colums] = scaler.transform(finaltestog[colums])

    ######################## RUL creation test #################################
    rul = pd.DataFrame(test_dataog.groupby('id')['cycle'].max()).reset_index()
    rul.columns = ['id', 'max']
    truth.columns = ['more']

    truth['id'] = truth.index + 1
    truth['max'] = rul['max'] + truth['more']

    truth.drop('more', axis=1, inplace=True)

    test_dataog = test_dataog.merge(truth, on=['id'], how='left')  # merge truth to include it

    test_dataog['RUL'] = test_dataog['max'] - test_dataog['cycle']

    test_dataog.drop('max', axis=1, inplace=True)
    ########################test normalization################################

    test_dataog['cycle_norm'] = test_dataog['cycle']
    test_dataog[colums] = scaler.transform(test_dataog[colums])

    train_dataog.to_csv('normalizedtrain_' + datalist + '.csv', index=None)
    test_dataog.to_csv('normalizedtest_' + datalist + '.csv', index=None)

    ################################drop appropriate columns#######################
    if datalist == 'FD001' or 'FD003':
        train_dataog = train_dataog.drop('s1', axis=1)
        test_dataog = test_dataog.drop('s1', axis=1)
        finaltestog = finaltestog.drop('s1', axis=1)
        train_dataog = train_dataog.drop('setting3', axis=1)
        test_dataog = test_dataog.drop('setting3', axis=1)
        finaltestog = finaltestog.drop('setting3', axis=1)
        train_dataog = train_dataog.drop('s5', axis=1)
        test_dataog = test_dataog.drop('s5', axis=1)
        finaltestog = finaltestog.drop('s5', axis=1)
        train_dataog = train_dataog.drop('s18', axis=1)
        test_dataog = test_dataog.drop('s18', axis=1)
        finaltestog = finaltestog.drop('s18', axis=1)
        train_dataog = train_dataog.drop('s16', axis=1)
        test_dataog = test_dataog.drop('s16', axis=1)
        finaltestog = finaltestog.drop('s16', axis=1)
        train_dataog = train_dataog.drop('s19', axis=1)
        test_dataog = test_dataog.drop('s19', axis=1)
        finaltestog = finaltestog.drop('s19', axis=1)
    if datalist == 'FD001':
        train_dataog = train_dataog.drop('s10', axis=1)
        test_dataog = test_dataog.drop('s10', axis=1)
        finaltestog = finaltestog.drop('s10', axis=1)
    ##################################create main file structure####################
    try:
        os.mkdir('figuressklearn')
    except OSError:
        pass


    train_data = copy.copy(train_dataog)
    test_data = copy.copy(test_dataog)
    finaltest = copy.copy(finaltestog)
    id_array = test_data['id']
    ################################# one hot-encode#####################################################
    '''    
    categoricalColumns = ['id']
    test_data = pd.get_dummies(test_data, prefix=['engine'], columns=categoricalColumns)
    train_data = pd.get_dummies(train_data, prefix=['engine'], columns=categoricalColumns)
    finaltest = pd.get_dummies(finaltest, prefix=['engine'], columns=categoricalColumns)
    '''
    columnlist = [('test_data', len(test_data.columns)), ('train_data', len(train_data.columns)),
                  ('final_test', len(finaltest.columns))]  # create a key:val tuple list

    columnlist = sorted(columnlist, reverse=True, key=lambda x: x[1])  # sort tuple list

    column1diff = columnlist[0][1] - columnlist[1][1]
    column2diff = columnlist[0][1] - columnlist[2][1]

    if columnlist[1][0] == 'test_data':
        for col in range(column1diff):
            test_data[col] = 0
        for col in range(column2diff):
            train_data[col] = 0  # pad columns on each to equalize columns
    elif columnlist[1][0] == 'train_data':
        for col in range(column1diff):
            train_data[col] = 0
        for col in range(column2diff):
            test_data[col] = 0
    elif columnlist[1][0] == 'final_test':
        print('final list has not the max columns')
        exit()
    columnlist = [('test_data', len(test_data.columns)), ('train_data', len(train_data.columns)),
                  ('final_test', len(train_data.columns))]


    train_data = train_data.drop('cycle', axis=1)# drop cycles now that they have no use as cycle_num exists
    test_data = test_data.drop('cycle', axis=1)
    finaltest = finaltest.drop('cycle', axis=1)


    train_data.loc[train_data.RUL > 127, 'RUL'] = 127# limit RUL to 127
    test_data.loc[test_data.RUL > 127, 'RUL'] = 127
    finaltest.loc[finaltest.RUL > 127, 'RUL'] = 127
    train_labels = train_data['RUL']
    test_labels = test_data['RUL']
    final_labels = finaltest['RUL']


    train_data = train_data.drop('RUL', axis=1)# remove RUL from dataset
    test_data = test_data.drop('RUL', axis=1)
    finaltest = finaltest.drop('RUL', axis=1)



    N_train = len(train_data.columns)
    N_test = len(test_data.columns)

    if N_train != N_test:
        print("Datasets have incompatible column counts: %d vs %d" % (N_train, N_test))
        exit()
    M_train = len(train_data)
    M_test = len(test_data)


    test_labels = test_labels.to_numpy()
    train_labels = train_labels.to_numpy()
    train_labels.shape = (-1, 1)
    test_labels.shape = (-1, 1)

    lreg = LinearRegression()
    lregPoly2 = LinearRegression()
    lregpoly3 = LinearRegression()
    svrreg = svm.TimeSeriesSVR()

    poly2 = PolynomialFeatures(degree=2)
    poly3 = PolynomialFeatures(degree=3)

    x_trainP = poly2.fit_transform(train_data)
    x_testP = poly2.fit_transform(test_data)
    y_trainP = poly2.fit_transform(train_labels)
    y_testP=poly2.fit_transform(test_labels)
    x_trainP3 = poly3.fit_transform(train_data)
    x_testP3 = poly3.fit_transform(test_data)
    y_trainP3 = poly3.fit_transform(train_labels)
    y_testP3 = poly3.fit_transform(test_labels)
    test_datacopy=copy.copy(test_data)
    test_labelscopy=copy.copy(test_labels)


    lreg.fit(train_data, train_labels)
    lregPoly2.fit(x_trainP, y_trainP)
    lregpoly3.fit(x_trainP3, y_trainP3)
    #svrreg.fit(train_data, train_labels.ravel())
    print(' finished fit for '+datalist)
    lregpred = lreg.predict(test_data)
    #svrpred = svrreg.predict(test_data)
    poly2pred = lregPoly2.predict(x_testP)
    poly3pred = lregpoly3.predict(x_testP3)
    predictions = {}
    predictions['linear regression'] = lregpred
    #predictions['Time series SVR] = svrpred
    predictions['polynomial regression order 2'] = poly2pred
    predictions['polynomial regression order 3'] = poly3pred

    lregmse = mean_squared_error(test_labels, lregpred)
    #svrmse = mean_squared_error(test_labels, svrpred)
    poly2mse = mean_squared_error(y_testP, poly2pred)
    poly3mse = mean_squared_error(y_testP3, poly3pred)

    lregmae = mean_absolute_error(test_labels, lregpred)
    #svrmae = mean_absolute_error(test_labels, svrpred)
    poly2mae = mean_absolute_error(y_testP, poly2pred)
    poly3mae = mean_absolute_error(y_testP3, poly3pred)

    lregmape = mean_absolute_percentage_error(test_labels, lregpred)
    #svrmape = mean_absolute_percentage_error(test_labels, svrpred)
    poly2mape = mean_absolute_percentage_error(y_testP, poly2pred)
    poly3mape = mean_absolute_percentage_error(y_testP3, poly3pred)

    lregr2 = r2_score(test_labels, lregpred)
    #svrr2 = r2_score(test_labels, svrpred)
    poly2r2 = r2_score(y_testP, poly2pred)
    poly3r2 = r2_score(y_testP3, poly3pred)

    lregmetrics = {}
    lregmetrics['mse'] = lregmse
    lregmetrics['rmse'] = np.sqrt(lregmse)
    lregmetrics['mae'] = lregmae
    lregmetrics['mape'] = lregmape
    lregmetrics['r2score'] = lregr2


    #svrmetrics = {}
    #svrmetrics['mse'] = svrmse
    #svrmetrics['mae'] = svrmae
    #svrmetrics['mape'] = svrmape
    #svrmetrics['r2score'] = svrr2

    poly2metrics = {}
    poly2metrics['mse'] = poly2mse
    poly2metrics['rmse'] = np.sqrt(poly2mse)
    poly2metrics['mae'] = poly2mae
    poly2metrics['mape'] = poly2mape
    poly2metrics['r2score'] = poly2r2

    poly3metrics = {}
    poly3metrics['mse'] = poly3mse
    poly3metrics['rmse'] = np.sqrt(poly3mse)
    poly3metrics['mae'] = poly3mae
    poly3metrics['mape'] = poly3mape
    poly3metrics['r2score'] = poly3r2

    innerdict['linear regression metrics '] = lregmetrics
    innerdict['polynomial regression metrics of order 2 '] = poly2metrics
    innerdict['polynomial regression metrics of order 3 '] = poly3metrics
    #innerdict['Time series SVR regression metrics'] = svrmetrics

    predictions['linear regression'] = lregpred
    #predictions['Time series SVR] = svrpred
    predictions['polynomial regression order 2'] = poly2pred
    predictions['polynomial regression order 3']
    outerdict[datalist] = innerdict
##################################run through results for plotting######################################################
    for key, prediction in predictions.items():
        if key == 'linear regression':
            test_data = test_datacopy
            test_labels = test_labelscopy
        elif key == 'polynomial regression order 2':
            test_data = x_testP
            test_labels = y_testP
        elif key == 'polynomial regression order 3':
            test_data = x_testP3
            test_labels = y_testP3
        for id in range(id_array.max()):
            id = id + 1

            indexes = np.where(test_data == id)[0]

            newplot = prediction[indexes]
            newtest = test_labels[indexes]

            testRMSEpereng = np.sqrt(mean_squared_error(newplot, newtest))
            try:
                os.mkdir('figuressklearn/Model={T} Datalist={e}'.format(T=str(key), e=datalist))
            except OSError:
                pass
############################################produce sole engine predictions graph#######################################
            pyplot.plot(newplot)
            pyplot.plot(newtest)
            pyplot.legend(['prediction', 'reality'], loc='upper right')
            pyplot.title('prediction vs reality' + ' Model=' + str(key) + ', Datalist=' + datalist + ', ID=' + str(id))
            pyplot.xlabel('Cycle')
            pyplot.ylabel('RUL')
            pyplot.savefig('figuressklearn/Model={T} Datalist={e}/ID={id}.png'.format(T=str(key), e=datalist, id=str(id)))
            pyplot.clf()
###########################################produce cycles concatenated##################################################
        pyplot.plot(prediction)
        pyplot.plot(test_labels)
        pyplot.legend(['prediction', 'reality'], loc='upper right')
        pyplot.title('prediction vs reality' + ' Model=' + str(key) + ', Datalist=' + datalist)
        pyplot.xlabel('Cycles concatenated')
        pyplot.ylabel('RUL')
        pyplot.savefig('figuressklearn/Model={T} Datalist={e}/all.png'.format(T=str(key), e=datalist))
        pyplot.show()
        datacount = 0
        score = 0

pprint.pprint(outerdict)
pprint.pprint(outerdict, stream=open("resultsklearn.txt", 'w+'))

