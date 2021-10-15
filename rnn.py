import pandas as pd
import copy
import numpy as np

import tensorflow as tf
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error
from matplotlib import pyplot
from sklearn.preprocessing import LabelEncoder
import os
import pprint
from sklearn import preprocessing
import math

datalists = ['FD001', 'FD002', 'FD003', 'FD004']
label_encoder = LabelEncoder()
Timestamps = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
epochs = [200]

outerresdict = {}
############################import datasets################################################

for datalist in datalists:  # loops over each dataset
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
        os.mkdir('models')
    except OSError:
        pass
    try:
        os.mkdir('figures')
    except OSError:
        pass
    try:
        os.mkdir('logs')
    except OSError:
        pass
    ######################################loop over Timestamps##############################

    for T in Timestamps:
        innerresdict = {}


        train_data = copy.copy(train_dataog)
        test_data = copy.copy(test_dataog)
        finaltest = copy.copy(finaltestog)



        for id in train_data['id'].unique():  # drop enough rows from the start per engine so that shape is uniform
            maxval = train_data.loc[(train_data['id'] == id)]['RUL'].max()
            minval = train_data.loc[(train_data['id'] == id)]['RUL'].min()
            maxvallocation = train_data.loc[(train_data['id'] == id)]['RUL'].idxmax()
            diff = 1 + maxval - minval

            VALS_TO_REMOVE = ((diff) % T)

            for i in range(VALS_TO_REMOVE):
                remove = i + maxvallocation
                train_data = train_data.drop(remove, axis=0)
            maxvallocation = train_data.loc[(train_data['id'] == id)]['RUL'].idxmax()
            minvallocation = train_data.loc[(train_data['id'] == id)]['RUL'].idxmin()

        for id in test_data['id'].unique():  # done for test as well
            maxval = test_data.loc[(test_data['id'] == id)]['RUL'].max()
            minval = test_data.loc[(test_data['id'] == id)]['RUL'].min()
            maxvallocation = test_data.loc[(test_data['id'] == id)]['RUL'].idxmax()
            diff = 1 + maxval - minval

            VALS_TO_REMOVE = ((diff) % T)

            if VALS_TO_REMOVE == 0:
                continue

            for i in range(int(VALS_TO_REMOVE)):
                remove = i + maxvallocation
                test_data = test_data.drop(remove, axis=0)
            maxvallocation = test_data.loc[(test_data['id'] == id)]['RUL'].idxmax()
            minvallocation = test_data.loc[(test_data['id'] == id)]['RUL'].idxmin()

        for id in finaltest.id.unique():  # all actions repeated for finaltest so that a score would have been produced
            maxval = finaltest.loc[(finaltest['id'] == id)]['RUL'].max()
            minval = finaltest.loc[(finaltest['id'] == id)]['RUL'].min()
            maxvallocation = finaltest.loc[(finaltest['id'] == id)]['RUL'].idxmax()
            diff = 1 + maxval - minval

            VALS_TO_REMOVE = ((diff) % T)

            for i in range(VALS_TO_REMOVE):
                remove = i + maxvallocation
                finaltest = finaltest.drop(remove, axis=0)
            maxvallocation = finaltest.loc[(finaltest['id'] == id)]['RUL'].idxmax()
            minvallocation = finaltest.loc[(finaltest['id'] == id)]['RUL'].idxmin()
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

        train_data = train_data.drop('cycle', axis=1)  # drop cycles now that they have no use as cycle_num exists
        test_data = test_data.drop('cycle', axis=1)
        finaltest = finaltest.drop('cycle', axis=1)

        train_data.loc[train_data.RUL > 127, 'RUL'] = 127  # limit RUL to 127
        test_data.loc[test_data.RUL > 127, 'RUL'] = 127
        finaltest.loc[finaltest.RUL > 127, 'RUL'] = 127
        train_labels = train_data['RUL']
        test_labels = test_data['RUL']
        final_labels = finaltest['RUL']

        train_data = train_data.drop('RUL', axis=1)  # remove RUL from dataset
        test_data = test_data.drop('RUL', axis=1)
        finaltest = finaltest.drop('RUL', axis=1)

        N_train = len(train_data.columns)
        N_test = len(test_data.columns)

        if N_train != N_test:
            print("Datasets have incompatible column counts: %d vs %d" % (N_train, N_test))
            exit()
        M_train = len(train_data)
        M_test = len(test_data)

        # convert to numpy for data re-shaping
        train_data = train_data.to_numpy()
        test_data = test_data.to_numpy()
        finaltest = finaltest.to_numpy()
        train_labels = train_labels.to_numpy()
        test_labels = test_labels.to_numpy()
        final_labels = final_labels.to_numpy()
        id_arraynp = id_array.to_numpy()

        columns = train_data.shape[1]

        test_data.shape = (-1, T, columns)
        train_data.shape = (-1, T, columns)
        finaltest.shape = (-1, T, columns)
        train_labels.shape = (-1, T, 1)
        test_labels.shape = (-1, T, 1)
        final_labels.shape = (-1, T, 1)

        id_arraynp.shape = (-1, T, 1)

        for e in epochs:
            try:
                os.mkdir('models/T={T}_Datalist={e}'.format(T=str(T), e=datalist))
            except OSError:
                pass

            checkpoint = tf.keras.callbacks.ModelCheckpoint(
                filepath='models/T={T}_Datalist={e}'.format(T=str(T), e=datalist), save_best_only=True,
                save_weights_only=True
            )
            tensorboard = tf.keras.callbacks.TensorBoard(log_dir='logs/T={T}_Datalist={e}'.format(T=str(T), e=datalist))
            earlystopping = tf.keras.callbacks.EarlyStopping(min_delta=5, patience=10)
            model = tf.keras.models.Sequential([
                tf.keras.layers.LSTM(units=100, input_shape=(T, N_train), return_sequences=True),
                tf.keras.layers.Dropout(0.2),
                tf.keras.layers.LSTM(units=300, return_sequences=True),
                tf.keras.layers.Dropout(0.2),
                tf.keras.layers.LSTM(units=200, return_sequences=True),
                tf.keras.layers.Dropout(0.2),
                tf.keras.layers.LSTM(units=100, return_sequences=False),

                tf.keras.layers.Dense(1)
            ])
            model.compile(optimizer='adam',
                          loss='mse',
                          metrics=['mae', 'mape'])  # how to add rsquared   tfa.metrics.RSquare(y_shape=(1, ))
            model.summary()
            try:
                os.mkdir('figures/T={T}_Datalist={e}'.format(T=str(T), e=datalist))
            except OSError:
                pass

            history = model.fit(train_data, train_labels, epochs=e, validation_data=(test_data, test_labels),
                                callbacks=[checkpoint, tensorboard, earlystopping])

            model.load_weights('models/T={T}_Datalist={e}'.format(T=str(T), e=datalist))

            res = model.evaluate(test_data, test_labels)
            pred_train = model.predict(train_data)
            # get labels in useful form
            test_labelscopy = copy.copy(test_labels[:, 0, :])
            train_labelscopy = copy.copy(train_labels[:, 0, :])
            final_labelscopy = copy.copy(final_labels[:, 0, :])
            id_arraynpcopy = id_arraynp[:, 0, :]

            trainRMSE = np.sqrt(mean_squared_error(train_labelscopy, pred_train))
            pred = model.predict(test_data)
            finalpred = model.predict(finaltest)
            testRMSE = np.sqrt(mean_squared_error(test_labelscopy, pred))
            if optimaltimestep > testRMSE:
                optimaltimestep = testRMSE
                besttimestep = True
            else:
                besttimestep = False
#####################get best metrics per loop and produce training graph###############################################
            bestmse = mean_squared_error(test_labelscopy, pred)
            bestlistmse.append(bestmse)
            innerresdict['mse'] = bestmse
            innerresdict['rmse'] = math.sqrt(bestmse)

            bestmae = mean_absolute_error(test_labelscopy, pred)
            bestlistmae.append(bestmae)
            innerresdict['mae'] = bestmae

            bestmape = mean_absolute_percentage_error(test_labelscopy, pred)
            bestlistmape.append(bestmape)
            innerresdict['mape'] = bestmape

            r2score = r2_score(test_labelscopy, pred)
            innerresdict['r2'] = r2score

            pyplot.plot(history.history['loss'])
            pyplot.plot(history.history['val_loss'])
            pyplot.title('model train vs loss' + ', T=' + str(T) + ', Datalist=' + datalist)
            pyplot.ylabel('loss')
            pyplot.xlabel('epoch')
            pyplot.legend(['train', 'validation'], loc='upper right')
            pyplot.savefig('figures/T={T}_Datalist={e}/line_graph.png'.format(T=str(T), e=datalist))
            pyplot.show()
            print(pred)
            test_datacopy = copy.copy(test_data[:, 0, :])

            test_datacopy = np.append(test_datacopy, id_arraynpcopy, axis=1)
            enginesskipped = 0
            dictmse = {}
######################calculate RMSEs per engine and plot sole engine predictions#######################################
            for id in range(id_array.max()):
                id = id + 1
                indexes = np.where(test_datacopy[:, len(test_datacopy[0, :]) - 1] == id)[0]

                newplot = pred[indexes]
                newtest = test_labelscopy[indexes]

                testRMSEpereng = np.sqrt(mean_squared_error(newplot, newtest))
                if T == Timestamps[0]:
                    rmsepereng[id] = testRMSEpereng
                else:
                    rmsepereng[id] = (testRMSEpereng + rmsepereng[id])
                dictmse[id] = testRMSEpereng
                if besttimestep == True:  # find a way to make an updating T and data so that best results is of min mse
                    bestrmsepereng[id] = testRMSEpereng

                pyplot.plot(newplot)
                pyplot.plot(newtest)
                pyplot.legend(['prediction', 'reality'], loc='upper right')
                pyplot.title('prediction vs reality' + ' T=' + str(T) + ', Datalist=' + datalist + ', ID=' + str(id))
                pyplot.xlabel('Cycle')
                pyplot.ylabel('RUL')
                pyplot.savefig('figures/T={T}_Datalist={e}/ID={id}.png'.format(T=str(T), e=datalist, id=str(id)))
                pyplot.clf()
##################################filter results by RMSE and plot cycles concatenated###################################
            for key, val in dictmse.items():
                if val < 20:
                    rmseperengskipped[key] = val
                else:
                    enginesskipped += 1
            rmsewithskipped = sum(rmseperengskipped.values()) / len(rmseperengskipped)

            innerresdict['rmsewithvals>20'] = rmsewithskipped
            innerresdict['total values more than 20'] = enginesskipped

            pyplot.plot(pred)
            pyplot.plot(test_labelscopy)
            pyplot.legend(['prediction', 'reality'], loc='upper right')
            pyplot.title('prediction vs reality' + ' T=' + str(T) + ', Datalist=' + datalist)
            pyplot.xlabel('Cycles concatenated')
            pyplot.ylabel('RUL')
            pyplot.savefig('figures/T={T}_Datalist={e}/all.png'.format(T=str(T), e=datalist))
            pyplot.show()
            datacount = 0
            score = 0
#################################calculating score if test set was complete##########################
            for datum in finalpred:

                d = final_labelscopy[datacount] - datum
                if d < 0:
                    indici = -(d / 10)
                    score = score + ((np.exp(indici)) - 1)
                else:
                    indici = -(d / 13)
                    score = score + ((np.exp(indici)) - 1)
                datacount += 1
            resultsdict[('T={T}_e={e}'.format(T=str(T), e=str(e)))] = (np.sqrt(res[0]), score[0])
            print((np.sqrt(res[1]), score[0]))
            outerresdict['T={T}_Datalist={e}'.format(T=str(T), e=datalist)] = innerresdict


        if besttimestep == True:
            listbestrmseeng = bestrmsepereng.items()
            ID, bestrmse = zip(*listbestrmseeng)
            pyplot.scatter(ID, bestrmse)
            pyplot.legend(['RMSE per engine'], loc='upper right')
            pyplot.title('Best RMSE per engine ' + datalist + ' T= ' + str(T))
            pyplot.xlabel('Engine ID')
            pyplot.ylabel('RMSE')
            pyplot.savefig('figures/BestRMSE_Datalist={e}.png'.format(T=str(T), e=datalist))
            pyplot.show()
    pprint.pprint(resultsdict)
###########################################create metrics vs Timesteps#######################################
    pyplot.plot(Timestamps, bestlistmse)
    pyplot.legend(['MSE'], loc='upper right')
    pyplot.title('MSE vs Timestep ' + datalist)
    pyplot.xlabel('Timestep')
    pyplot.ylabel('MSE')
    pyplot.savefig('figures/bestlistmse_Datalist={e}.png'.format(T=str(T), e=datalist))
    pyplot.show()

    pyplot.plot(Timestamps, bestlistmae)
    pyplot.legend(['MAE'], loc='upper right')
    pyplot.title('MAE vs Timestep ' + datalist)
    pyplot.xlabel('Timestep')
    pyplot.ylabel('MAE')
    pyplot.savefig('figures/bestlistmae_Datalist={e}.png'.format(T=str(T), e=datalist))
    pyplot.show()

    pyplot.plot(Timestamps, bestlistmape)
    pyplot.legend(['MAPE'], loc='upper right')
    pyplot.title('MAPE vs Timestep ' + datalist)
    pyplot.xlabel('Timestep')
    pyplot.ylabel('MAPE')
    pyplot.savefig('figures/bestlistmape_Datalist={e}.png'.format(T=str(T), e=datalist))
    pyplot.show()
######################################create-average RMSE per engine graph##############################
    for id in rmsepereng:
        rmsepereng[id] = rmsepereng[id] / len(Timestamps)
    listrmseeng = rmsepereng.items()
    ID, avgrmse = zip(*listrmseeng)
    pyplot.scatter(ID, avgrmse)

    pyplot.legend(['RMSE per engine'], loc='upper right')
    pyplot.title('avg RMSE per engine ' + datalist)
    pyplot.xlabel('Engine ID')
    pyplot.ylabel('RMSE')
    pyplot.savefig('figures/avgRMSE_Datalist={e}.png'.format(T=str(T), e=datalist))
    pyplot.show()
    outerresdict['T={T}_Datalist={e}'.format(T=str(T), e=datalist)] = innerresdict
    print(ID, avgrmse)
pprint.pprint(outerresdict)
pprint.pprint(outerresdict, stream=open("result.txt", 'w+'))
