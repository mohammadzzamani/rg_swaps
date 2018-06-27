import pandas as pd
import re
import matplotlib.pyplot as plt
import math
import numpy as np
from itertools import chain
import datetime
from sklearn.feature_extraction import DictVectorizer as DV
from sklearn import linear_model, svm
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn import metrics
import Util as U
import pickle



def get_ratio(data, year, month, date_col, label_col = 'test_status'):

    eval_min_date = datetime.date(year, month-2, 1)
    eval_max_date = datetime.date(year, month, 1)

    eval = data[data[date_col] >= eval_min_date]
    eval = eval[eval[date_col] < eval_max_date]

    # eval = data[data[date_col] < eval_max_date]

    zeros = eval[eval[label_col] == 0].shape[0]
    ones = eval[eval[label_col] == 1].shape[0]
    print ('zeros: ', zeros)
    print ('ones: ', ones)
    ratio = zeros*1.0/(zeros + ones)
    ratio = (ratio + 0.64)/2
    print ('eval ratio: ', ratio)

    return ratio



def split_train_test(data, date_col, test_date):
    # train_min_date = datetime.date(2017,04,01)
    train_max_date = datetime.date(test_date['year'],test_date['month'],test_date['day'])


    train = data[data[date_col] < datetime.date(2017,7,01)]
    print ('07_01: ', train.shape)
    train = data[data[date_col] < datetime.date(2017,8,01)]
    print ('08_01: ', train.shape)
    train = data[data[date_col] < datetime.date(2017,9,01)]
    print ('09_01: ', train.shape)
    train = data[data[date_col] < datetime.date(2017,10,01)]
    print ('10_01: ', train.shape)
    train = data[data[date_col] < datetime.date(2017,11,01)]
    print ('11_01: ', train.shape)


    test_min_date = datetime.date(test_date['year'],test_date['month'],test_date['day'])
    test_max_date = datetime.date(test_date['year'],test_date['month']+1,test_date['day'])


    # train = data[data[date_col] >= train_min_date]
    train = data[data[date_col] < train_max_date]

    test = data[data[date_col] >= test_min_date]
    test = test[test[date_col] < test_max_date]

    return [train , test]

def sampling(data, col_name, sample_rate=1.0):
    neg_data = data[data[col_name] == 0]
    print (neg_data.shape)
    neg_data = neg_data.sample(int(math.floor(sample_rate * neg_data.shape[0])))
    print (neg_data.shape)
    pos_data = data[data[col_name] == 1]
    print (pos_data.shape)
    data = neg_data.append(pos_data)
    print 'after sampling shape: ' , data.shape
    return data

def split_x_y(data, label_col = 'test_status', rgmodels_list= []):

    xcols = rgmodels_list + [ 'duration' , 'age_of_device' , 'use_factor' , 'rg_days_on_ban'  , 'calendar_day' , 'time_of_day'] #, 'tech_ntf_swap_ratio', 'manager_ntf_swap_ratio']
    # xcols = [ 'duration' , 'age_of_device' , 'use_factor' , 'rg_days_on_ban'  , 'calendar_day' , 'time_of_day']
    # xcols = ['first_rgmodel_5031NV-030', 'first_rgmodel_5268AC', 'first_rgmodel_NVG589', 'first_rgmodel_NVG599', 'time_spent_min', 'time_spent_hour', 'down' , 'todn' , 'rank_of_ban_current_is_biggest', 'rg_days_on_ban']
    # xcols = ['time_spent_min', 'time_spent_hour', 'down' , 'todn' , 'age_of_device', 'rank_of_ban_current_is_biggest', 'rg_days_on_ban']
    ycols = [label_col]

    xdf = data[xcols]
    ydf = data[ycols]


    zeros = ydf[ydf[label_col] == 0].shape[0]
    ones = ydf[ydf[label_col] == 1].shape[0]
    print ('zeros: ', zeros)
    print ('ones: ', ones)
    # epsilon = 0.025
    # ratio = -1 if zeros + ones == 0 else  zeros*1.0/(zeros + ones)
    # print ("ratio: %.2f"% ratio)

    # ratio = ratio * 0.75 + 0.25
    # ratio = 0.73
    # print ("ratio: %.2f"% ratio)

    return xdf, ydf



def predict_from_proba(proba, threshold):
    pred = np.array([ 0 if proba[i,0] > threshold else 1 for i in range(proba.shape[0])])
    return pred;


def measure(true, pred, day, month, report=False):

    print("Mean squared error: %.2f" % mean_squared_error(true, pred))
    print("Mean absolute error: %.2f" % mean_absolute_error(true, pred))
    print ('sum_pred: ', sum(pred))
    print ('sum_true: ', sum(true))
    conf_mat = confusion_matrix(true, pred)
    print ('counts: ' , sum(sum(conf_mat)))
    print ( str( conf_mat[1,0]*1.0 / (conf_mat[1,0] + conf_mat[0,0])) + ' vs ' +  str( conf_mat[1,1]*1.0 / (conf_mat[1,1] + conf_mat[0,1])))
    print (conf_mat)

    if report:
        with open(name = 'outputs/conf_mats.csv', mode='a') as myfile:
            myfile.write(str(month) + "-" + str(day))
            myfile.write(','+ str(conf_mat[0,0]) + ',' + str(conf_mat[0,1]))
            myfile.write(','+','+ str( round(conf_mat[1,0]*1.0 / (conf_mat[1,0] + conf_mat[0,0]),3)) + ' , ' +  str( round(conf_mat[1,1]*1.0 / (conf_mat[1,1] + conf_mat[0,1]),3))+'\n')
            myfile.write(','+ str(conf_mat[1,0]) + ',' + str(conf_mat[1,1])+'\n\n')

            # np.save(myfile, conf_mat)
            myfile.close()

    print "zeros ( false positive rate)  %.2f" %  (conf_mat[0,1]*1.0 / (conf_mat[0,0] + conf_mat[0,1]))
    print "ones: ( true positive rate, recall ) %.2f" % (conf_mat[1,1]*1.0 / (conf_mat[1,0] + conf_mat[1,1]))
    print("precision: %.2f" % metrics.precision_score(true, pred, average='binary'))
    print("recall: %.2f" % metrics.recall_score(true, pred, average='binary') )
    print("f1_score: %.2f" % metrics.f1_score(true, pred, average='binary'))


def simple_learning_model(xtrain, ytrain, xtest=None, ytest=None, ratio = 0.5, rg_model = '...', day='', month='', report=False, method='lr', pickle_filename=None):

    # clf = linear_model.SGDClassifier()
    # clf = RandomForestClassifier(max_depth=4, min_samples_split=10, min_samples_leaf=5, min_impurity_decrease=0.001, random_state=0)
    # clf = RandomForestClassifier(n_estimators = 100, max_depth=10, random_state=0)
    # clf = DecisionTreeClassifier()

    ypred_proba = None
    ypred = None
    if method == 'lr':
        clf = linear_model.LogisticRegression()
        clf.fit(xtrain, ytrain)
        if xtest is not None:
            ypred_proba = clf.predict_proba(xtest)
            ypred = predict_from_proba(ypred_proba, ratio)
    else:
        # clf = svm.SVC(C=0.0001, kernel='linear', gamma=100)
        clf = GradientBoostingClassifier(n_estimators=100, subsample=0.66, min_impurity_decrease = 0.001,max_features=6, max_depth=7)
        clf.fit(xtrain, ytrain)
        if xtest is not None:
            ypred = clf.predict(xtest)

    # params = {'cutoff': ratio}
    if pickle_filename:
        pickle.dump((clf, method, ratio, rg_model), open(pickle_filename, 'wb'))

    # print ypred_proba[:10,:]
    # print ypred[:10]
    # print ytest[:10]


    #
    # ypred = clf.predict(xtest)
    # # The coefficients
    # # print('Coefficients: \n', clf.coef_)
    # print ('ytest:')
    # print (ytest[:50].transpose())
    # print ('ypred:')
    # print (yypred[:50])
    # print (type(ypred))
    # print (type(ytest))

    if xtest is not None:
        print("{0}    {1}".format("<"*50, rg_model))
        measure(ytest, ypred, day=day, month=month, report=report)
        print("{0}".format(">"*50))
    else:
        print("{0}    {1} - train_only ".format("<"*50, rg_model))

    return ypred, ypred_proba


def simple_regression(xtrain, ytrain, xtest, ytest, ratio = 0.5, rg_model = '__rg_model__', model=''):

    reg = linear_model.LinearRegression()
    reg.fit(xtrain, ytrain)

    ypred= reg.predict(xtest)
    ypred_proba = np.hstack(((1-ypred) , ypred))
    ypred = predict_from_proba(ypred_proba, ratio)
    print ('ypred: ' , ypred[:10])
    print ('ytest: ', ytest[:10])



    #
    # ypred = clf.predict(xtest)
    # # The coefficients
    # # print('Coefficients: \n', clf.coef_)
    # print ('ytest:')
    # print (ytest[:50].transpose())
    # print ('ypred:')
    # print (yypred[:50])
    # print (type(ypred))
    # print (type(ytest))


    print("{0}    {1}".format("<"*50, rg_model))


    print ( 'ypred.shapes: ' , ypred.shape, ' , ytest.shape:  ' , ytest.shape)

    print("Mean squared error: %.2f" % mean_squared_error(ytest, ypred))
    print("Mean absolute error: %.2f" % mean_absolute_error(ytest, ypred))
    print ('sum_ypred: ', sum(ypred), 'max: ', max(ypred))
    print ('sum_ytest: ', sum(ytest))
    conf_mat = confusion_matrix(ytest, ypred)
    print (conf_mat)


    print "zeros ( false positive rate)  %.2f" %  (conf_mat[0,1]*1.0 / (conf_mat[0,0] + conf_mat[0,1]))
    print "ones: ( true positive rate, recall ) %.2f" % (conf_mat[1,1]*1.0 / (conf_mat[1,0] + conf_mat[1,1]))
    print("precision: %.2f" % metrics.precision_score(ytest, ypred, average='binary'))
    print("recall: %.2f" % metrics.recall_score(ytest, ypred, average='binary') )
    print("f1_score: %.2f" % metrics.f1_score(ytest, ypred, average='binary'))

    print("{0}".format(">"*50))

    return ypred


# def split_base_on_column(data, col_name):
#     data_dict = {}
#     values = data[col_name].unique().tolist()
#     for val in values:
#         print 'val: ', val
#         d = data[data[col_name] == val]
#
#         print d.shape
#         data_dict[val]  = d
#     return data_dict

def split_base_on_column(data, col_name):
    data_dict = {}
    values = data[col_name].unique().tolist()
    for val in values:
        data_model = data[data[col_name] == val]
        data_model.reset_index(drop=True, inplace=True)
        if data_model.shape[0] > 50000:
            print 'rg_model: ', val , ' count: ', data_model.shape[0]
            data_dict[val]  = data_model
    return data_dict

def prepare_data(data, separated_models = False, date_col='dt', label_col='test_status', model_col='model', dummy_encoding = False):


    data[date_col] =  pd.to_datetime(data[date_col], format='%Y-%m-%d %H:%M:%S.%f').dt.date


    #### convert category to nominal for label
    labels = ['PASS', 'FAIL']
    data = U.to_nominal(data,col_name=label_col, values=labels, nominal_col_name=label_col)

    #### convert time_of_day to nominal values
    data = U.to_nominal(data,col_name='time_of_day', nominal_col_name='time_of_day')
    # time_of_day_list = data['time_of_day'].unique().tolist()
    # data = pd.get_dummies(data,columns= ['time_of_day'], drop_first=False, prefix='', prefix_sep='')
    # calendar_day_list = [ 'day_'+ str(i) for i in data['calendar_day'].unique().tolist() ]
    # data = pd.get_dummies(data,columns= ['calendar_day'], drop_first=False, prefix='day', prefix_sep='_')

    # print ( 'dim: ', data.shape)
    # data = data[data['swap_to_receive_gap_days'] <= 45]
    # print ( 'dim: ', data.shape)
    #### convert time_spent_hour to int
    # data['time_spent_hour'] = data.time_spent_hour.map(lambda x: round(x,0))
    # data['time_spent_hour'] = data.time_spent_hour.astype(int)


    # data = data.replace('null', np.nan)
    # data.tech_ntf_swap_ratio = data.tech_ntf_swap_ratio.astype(float)
    # data.manager_ntf_swap_ratio = data.manager_ntf_swap_ratio.astype(float)
    # data['tech_ntf_swap_ratio'] = data.tech_ntf_swap_ratio.combine_first(data.manager_ntf_swap_ratio)
    # data['tech_ntf_swap_ratio'].fillna((data['tech_ntf_swap_ratio'].mean()), inplace=True)
    # data['manager_ntf_swap_ratio'].fillna((data['manager_ntf_swap_ratio'].mean()), inplace=True)

    # print ( data[data['tech_ntf_swap_ratio'].isnull()].ix[:100,:])
    #### filter rg_models
    # model_values = ['NVG599', 'NVG589', '5268AC', '5031NV-030']
    # data = data[data[model_col_name].isin(model_values)]

    U.report(data, 'nominalized labels')



    # separated_models = True
    #### sampling
    # sdf = sampling(sdf, 'trouble_code_desc_n', 0.1)
    if separated_models:

        #### convert category to nominal for rg_model
        data = U.to_nominal(data, model_col)

        #### split data based on rg_model
        data_dict = split_base_on_column(data, model_col)

        # data_dict = data.groupby(model_col)
        # print(data_dict.keys())
        # U.report(data_dict, 'data_dict:')
        rgmodels_list = []

    else:
        rgmodels_list = data[model_col].unique().tolist()

        #### dummy encoding of rg_models
        dummy_data = pd.get_dummies(data,columns= [model_col], drop_first=dummy_encoding, prefix='', prefix_sep='')
        data_dict = {'all_model': dummy_data}
        rgmodels_list = [i  for i in rgmodels_list if i in dummy_data.columns]

    rgmodels_list = rgmodels_list #+ time_of_day_list + calendar_day_list
    print ('rgmodels_list: ', rgmodels_list)
    return data_dict, rgmodels_list


def create_hist(data_dict, label_col = 'test_status', rgmodels_list = []):
    i = 0
    for key, data in data_dict.iteritems():
        print ( 'cols: ', data.columns)
        print (data.shape)
        [x, y, chert_] = split_x_y(data, label_col='test_status', rgmodels_list=rgmodels_list)
        for col in x.columns:
            plt.figure(i)
            i += 1
            # print col
            # print (x[col])
            # print 'trouble_code_desc_n: '
            # print (y['trouble_code_desc_n'])
            # plt.plot(x[col].values, y['trouble_code_desc_n'].values, 'ro')

            data1 = data[data[label_col] == 1]
            data0 = data[data[label_col] == 0]

            if col == 'calendar_day':
                fixed_bins = [0,1,2,3,4,5,6,7,8]
            elif col == 'time_of_day':
                fixed_bins = [0,1,2,3,4]
            elif col == 'duration':
                fixed_bins = [0,50,100,150,200,250,300,350]
            elif col == 'age_of_device':
                fixed_bins = [0,100, 200, 500, 1000, 2000, 5000]
            elif col == 'rg_days_on_ban':
                fixed_bins = [0,100, 200, 500, 1000, 2000, 5000]
            elif col == 'use_factor':
                fixed_bins = [1,2,3,4]
            else:
                continue

            n, bins, patches = plt.hist(data0[col].values, bins=fixed_bins, color='red', label='pass', alpha=0.5, normed=True)
            n, bins, patches = plt.hist(data1[col].values, bins=fixed_bins, color='blue', label='fail', alpha=0.5, normed=True)

            plt.legend(loc='upper right')
            plt.xlabel(col)
            plt.ylabel('test_status')
            # plt.title('About as simple as it gets, folks')
            plt.grid(True)
            plt.savefig('figures/'+key+'_'+col+'.png')
            # plt.show()


def print_detailed_outcomes(testdf, ypred, label_col='test_status', model_col='model', outputfilename=None):
    print ('shape: ', ypred.shape)
    testdf['ypred'] = ypred.reshape((len(ypred), 1))
    testdf['prediction'] = testdf['ypred'].map({0: 'NTF', 1:'TF'})

    if outputfilename is None:
        outputfilename = 'Y.csv'
    testdf.to_csv(outputfilename, sep=',')


    if model_col in testdf.columns:
        models= { m : testdf[testdf[model_col] == m ]for m in testdf[model_col].unique().tolist()}
    else:
        models = {}
        models['5268'] = testdf[testdf['5268AC'] ==1 ]
        models['589'] = testdf[testdf['NVG589'] ==1]
        models['599'] = testdf[testdf['NVG599'] ==1]
        models['5031'] = testdf[(testdf['5268AC'] == 0) & (testdf['NVG589'] == 0 ) & (testdf['NVG599'] == 0)]

    # models['5031'] = testdf[testdf['first_rgmodel'] == '5031NV-030']
    # models['5268'] = testdf[testdf['first_rgmodel'] == '5268AC']
    # models['589'] = testdf[testdf['first_rgmodel'] == 'NVG589']
    # models['599'] = testdf[testdf['first_rgmodel'] == 'NVG599']


    for key, val in models.iteritems():
        conf_mat = confusion_matrix(val[label_col], val['ypred'])
        recall = metrics.recall_score(val[label_col], val['ypred'])
        print ('model: ', key)
        print ('recall: ', recall )
        print ('counts: ' , sum(sum(conf_mat)))
        print conf_mat
        print ('--------')


def prepare_output(data, label, prediction, prediction_proba, ratio, label_col='test_status', prediction_col = 'prediction' , confidence_col = 'confidence', rood_dir = 'outputs'):

    data[label_col] = label
    data[prediction_col] = prediction
    prediction_proba = prediction_proba.round(decimals=2)
    data['passing_test_proba'] = prediction_proba[:,0]
    # data['failing_test_proba'] = prediction_proba[:,1]



    data[label_col] = data[label_col].apply(lambda x: 'PASS' if x == 0 else 'FAIL')
    data[prediction_col] = data[prediction_col].apply(lambda x: 'PASS' if x == 0 else 'FAIL')
    # data[confidence_col] = data.apply(lambda x: (x[prediction_proba_col] - ratio) / (1.0-ratio) if x[prediction_col]=='FAIL'
    #                                 else (ratio - x[prediction_proba_col]) / (ratio) )

    # U.report(data, 'output_data')
    data_dict= {}
    data_dict['L(FAIL)_P(FAIL)'] = data[(data[label_col]=='FAIL') & ( data[prediction_col] == 'FAIL')]
    data_dict['L(FAIL)_P(PASS)'] = data[(data[label_col]=='FAIL') & ( data[prediction_col] == 'PASS')]
    data_dict['L(PASS)_P(FAIL)'] = data[(data[label_col]=='PASS') & ( data[prediction_col] == 'FAIL')]
    data_dict['L(PASS)_P(PASS)'] = data[(data[label_col]=='PASS') & ( data[prediction_col] == 'PASS')]

    for key, value in data_dict.iteritems():
        value.to_csv(rood_dir+'/'+key, sep=',', index=False)





#### read whole data
# sdf = pd.read_table('/Users/Mz/Downloads/att_files/create_table/tf_label_201804_0614.csv', sep='\t')
sdf = pd.read_table('/Users/Mz/Downloads/att_files/create_table/tf_label_201804_0525-17.csv', sep='\t')


date_col, label_col, model_col ='swap_date' , 'test_status', 'model'

sdf_dict, rgmodels_list = prepare_data(sdf, separated_models=False, date_col=date_col, label_col=label_col, model_col=model_col)

U.report(sdf_dict, 'prepared_data')

# create_hist(sdf_dict, label_col='test_status', rgmodels_list=rgmodels_list)

for key, sdf in sdf_dict.iteritems():
    print 'rg_model: ' , key


    # for test_day in range(31,32):
    test_date ={'year':2018, 'month':3, 'day':1}

    #### split data to train & test
    [train, test] = split_train_test(sdf, date_col, test_date=test_date)


    #### split train & test to x and y
    xtrain, ytrain = split_x_y(train, label_col=label_col, rgmodels_list=rgmodels_list)
    xtest, ytest  = split_x_y(test, label_col=label_col, rgmodels_list=rgmodels_list)
    xall, yall  = split_x_y(test, label_col=label_col, rgmodels_list=rgmodels_list)


    #### get ratio from the last month of train
    ratio_train = get_ratio(sdf,year=test_date['year'], month=test_date['month'], date_col=date_col,label_col='test_status')
    # ratio_all = get_ratio(sdf,year=test_date['year'], month=test_date['month'], date_col=date_col,label_col='test_status')
    # ratio_train = ratio_all


    print 'shapes:'
    print xtrain.shape
    print xtest.shape
    print xtrain.columns



    #### Create linear regression model
    print ( 'train_columns: ', xtrain.columns)
    method = 'lr'
    pickle_filename ='outputs/pickles/'+str(method)+'_'+str(test_date['month']+1)+'.pickle'

    ypred , ypred_proba = simple_learning_model(xtrain.values, ytrain.values, xtest.values, ytest.values, ratio=ratio_train, rg_model=key, day=test_date['day'], month=test_date['month'], report=True, method=method)
    resultdf = print_detailed_outcomes(test, ypred, label_col=label_col)

    ypred , ypred_proba = simple_learning_model(xtrain.values, ytrain.values, xtrain.values, ytrain.values, ratio=ratio_train, rg_model=key, day=test_date['day'], month=test_date['month'], report=False, method=method)
    resultdf = print_detailed_outcomes(train, ypred, label_col=label_col)
    # simple_learning_model(xall.values, yall.values, ratio=ratio_all, rg_model= key, day=test_date['day'], month=test_date['month'], report=True, method=method, pickle_filename=pickle_filename)

    # resultdf = print_detailed_outcomes(test, ypred, label_col=label_col)
    # test_proba = ypred_proba[:,0]
    # test_pred = ypred
    #
    # ypred, ypred_proba = simple_learning_model(xtrain.values, ytrain.values, xtrain.values, ytrain.values, ratio , key)
    # train_proba = ypred_proba[:,0]
    # train_pred = ypred
    #
    # proba_col = 'proba'
    # pred_col='pred'
    # rgmodels_list = rgmodels_list + [proba_col, pred_col]
    # xtrain[proba_col] = train_proba
    # xtrain[label_col] = ytrain
    # xtrain[pred_col] = train_proba
    # xtest[proba_col] = test_proba
    # xtest[pred_col] = test_proba
    # xtest[label_col] = ytest
    #
    # test = xtest
    # train = U.upsample(xtrain, label_col=label_col)
    # xtrain, ytrain, ratio = split_x_y(train, label_col=label_col, rgmodels_list=rgmodels_list)
    # xtest, ytest, _chert_  = split_x_y(test, label_col=label_col, rgmodels_list=rgmodels_list)
    #
    #
    # ypred , ypred_proba = simple_learning_model(xtrain.values, ytrain.values, xtest.values, ytest.values, ratio, key, day=test_day, month=3, report=True, method='gb')


    # resultdf = print_detailed_outcomes(test, ypred, label_col=label_col)


    # xtest.insert(0, 'rg_serial_no', test['rg_serial_no'])
    # xtest.insert(1, 'ban', test['ban'])
    # prepare_output(xtest,ytest.values, ypred, ypred_proba, ratio)


    #### testing new method from here:
    # ypred, ypred_proba = simple_learning_model(xtrain.values, ytrain.values, xtrain.values, ytrain.values, (ratio - 0.25) / 0.75 , key)
    # resultdf = print_detailed_outcomes(train, ypred)
    # print ' =================================================================='

    # ytrain['NTF_prob'] = ypred_proba[:,0]
    # ytrain['TF_prob'] = ypred_proba[:,1]
    # def func(row):
    #     class_weight = 0.67
    #     return class_weight * row['trouble_code_desc_n'] + (1-class_weight) * row['TF_prob']
    #
    # ytrain['refined_label'] = ytrain.apply( func , axis=1)
    # print (ytrain.ix[:100,:])
    #
    # new_ytrain = ytrain[['refined_label']]
    # print( 'columns: ', ytrain.columns, ' ----- ' , new_ytrain.columns)
    # print ('shapes: ', ytrain.shape, '  ----- ' , new_ytrain.shape)
    # print ('new_train: ', new_ytrain.ix[:20,:])
    # ypred = simple_regression(xtrain.values, new_ytrain.values, xtest.values, ytest.values, ratio= ratio * 0.95, rg_model=key)
    #
    # # resultdf = print_detailed_outcomes(test, ypred)



