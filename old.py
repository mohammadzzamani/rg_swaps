import pandas as pd
import re
import matplotlib.pyplot as plt
import math
import numpy as np
from itertools import chain
import datetime
from sklearn.feature_extraction import DictVectorizer as DV
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn import metrics
import Util as U




def split_train_test(data, date_col):
    # train_min_date = datetime.date(2017,11,01)
    train_max_date = datetime.date(2017,11,01)
    test_max_date = datetime.date(2018,01,01)

    # swap_datetime_col = 'swap_datetime'


    # train = data[data[date_col] > train_min_date]
    train = data[data[date_col] < train_max_date]

    test = data[data[date_col] >= train_max_date]
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

def split_x_y(data):

    xcols = ['first_rgmodel_5031NV-030', 'first_rgmodel_5268AC', 'first_rgmodel_NVG589', 'first_rgmodel_NVG599' , 'time_spent_min', 'time_spent_hour', 'down' , 'todn' , 'age_of_device', 'rank_of_ban_current_is_biggest', 'rg_days_on_ban']
    # xcols = ['first_rgmodel_5031NV-030', 'first_rgmodel_5268AC', 'first_rgmodel_NVG589', 'first_rgmodel_NVG599', 'time_spent_min', 'time_spent_hour', 'down' , 'todn' , 'rank_of_ban_current_is_biggest', 'rg_days_on_ban']
    # xcols = ['time_spent_min', 'time_spent_hour', 'down' , 'todn' , 'age_of_device', 'rank_of_ban_current_is_biggest', 'rg_days_on_ban']
    ycols = ['trouble_code_desc_n']

    xdf = data[xcols]
    ydf = data[ycols]

    zeros = ydf[ydf.trouble_code_desc_n == 0].shape[0]
    ones = ydf[ydf.trouble_code_desc_n == 1].shape[0]
    print ('zeros: ', zeros)
    print ('ones: ', ones)
    ratio = zeros*1.0/(zeros + ones)
    print ("ratio: %.2f"% ratio)
    ratio = ratio * 0.9 + 0.1
    print ("ratio: %.2f"% ratio)

    return xdf, ydf, ratio



def predict_from_proba(proba, threshold):
    pred = np.array([ 0 if proba[i,0] > threshold else 1 for i in range(proba.shape[0])])
    return pred;


def measure(true, pred):

    print("Mean squared error: %.2f" % mean_squared_error(true, pred))
    print("Mean absolute error: %.2f" % mean_absolute_error(true, pred))
    print ('sum_pred: ', sum(pred))
    print ('sum_true: ', sum(true))
    conf_mat = confusion_matrix(true, pred)
    print ('counts: ' , sum(sum(conf_mat)))
    print (conf_mat)



    print "zeros ( false positive rate)  %.2f" %  (conf_mat[0,1]*1.0 / (conf_mat[0,0] + conf_mat[0,1]))
    print "ones: ( true positive rate, recall ) %.2f" % (conf_mat[1,1]*1.0 / (conf_mat[1,0] + conf_mat[1,1]))
    print("precision: %.2f" % metrics.precision_score(true, pred, average='binary'))
    print("recall: %.2f" % metrics.recall_score(true, pred, average='binary') )
    print("f1_score: %.2f" % metrics.f1_score(true, pred, average='binary'))


def simple_learning_model(xtrain, ytrain, xtest, ytest, ratio = 0.5, rg_model = '...'):

    # clf = linear_model.SGDClassifier()
    clf = linear_model.LogisticRegression()
    # clf = RandomForestClassifier(max_depth=4, min_samples_split=50, min_samples_leaf=5, min_impurity_decrease=0.005, random_state=0)
    # clf = RandomForestClassifier(n_estimators = 100, max_depth=10, random_state=0)
    # clf = DecisionTreeClassifier()
    clf.fit(xtrain, ytrain)

    ypred_proba = clf.predict_proba(xtest)
    ypred = predict_from_proba(ypred_proba, ratio)

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

    print("{0}    {1}".format("<"*50, rg_model))
    measure(ytest, ypred)
    print("{0}".format(">"*50))

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

def prepare_data(data, date_col='dt'):


    data[date_col] =  pd.to_datetime(data['end_time_local'], format='%Y-%m-%d %H:%M:%S.%f').dt.date

    last_rg_dt = 'last_rg_dt'
    last_rg_ban_dt = 'last_rg_ban_dt'
    data[last_rg_ban_dt] =  pd.to_datetime(data['last_rg_ban_effective_ts'], format='%Y-%m-%d %H:%M:%S.%f').dt.date
    data[last_rg_dt] =  pd.to_datetime(data['last_rg_effective_ts'], format='%Y-%m-%d %H:%M:%S.%f').dt.date

    data['datediff'] = data[last_rg_dt] - data[last_rg_ban_dt]
    # data['datadiff'] = today_date - data[date_col]

    # def date_dif(row, date_col='dt'):
    #     today_date = datetime.date(2018,02,01)
    #     return today_date - row[date_col]
    # data['datediff'] = data.apply(date_dif, date_col = date_col,  axis=1)

    data['datediff'] = data['datediff']/ np.timedelta64(1, 'D')
    data['age_of_device'] = data['age_of_device'] - data['datediff']
    # data['age_of_device_updated'] = data.apply(lambda row: row['age_of_device'] -  row[last_rg_dt] + row[swap_datetime_col], axis=1 )
    # df['Value'] = df.apply(lambda row: my_test(row['a'], row['c']), axis=1)

    print data.ix[:10,:]


    #### convert category to nominal for label
    trouble_codes = ['NTF', 'TF']
    data = U.to_nominal(data,'trouble_code_desc', trouble_codes)

    #### convert time_spent_hour to int
    data['time_spent_hour'] = data.time_spent_hour.map(lambda x: round(x,0))
    data['time_spent_hour'] = data.time_spent_hour.astype(int)

    model_col_name='first_rgmodel'
    model_values = ['NVG599', 'NVG589', '5268AC', '5031NV-030']
    data = data[data[model_col_name].isin(model_values)]

    print ('data: ')
    print (data.ix[:10,:])

    ''' '''
    dummy_data = pd.get_dummies(data,columns= [model_col_name])

    print ('dummy_data:')
    print dummy_data.ix[:10,:]


    data_dict= {}
    data_dict['all_model'] = dummy_data

    '''
    # #### convert category to nominal for rg_model
    data = to_nominal(data,'first_rgmodel')

    #### split data based on rg_model
    col_name='first_rgmodel'
    data_dict = split_base_on_column(data, col_name)
    '''

    #### sampling
    # sdf = sampling(sdf, 'trouble_code_desc_n', 0.1)
    return data_dict


def create_hist(data_dict):
    i = 0
    for key, data in data_dict.iteritems():
        print ( 'cols: ', data.columns)
        print (data.shape)
        [x, y] = split_x_y(data)
        for col in x.columns:
            plt.figure(i)
            i += 1
            # print col
            # print (x[col])
            # print 'trouble_code_desc_n: '
            # print (y['trouble_code_desc_n'])
            # plt.plot(x[col].values, y['trouble_code_desc_n'].values, 'ro')

            data1 = data[data['trouble_code_desc_n'] == 1]
            data0 = data[data['trouble_code_desc_n'] == 0]

            if col == 'down':
                fixed_bins = [1,2,3,4,5,6,7,8]
            elif col == 'todn':
                fixed_bins = [1,2,3,4]
            elif col == 'time_spent_hour':
                fixed_bins = [1,2,3,4,5,6]
            elif col == 'time_spent_min':
                fixed_bins = [50,100,150,200,250,300,350]
            elif col == 'age_of_device':
                fixed_bins = [100, 200, 500, 1000, 2000, 5000]
            elif col == 'rg_days_on_ban':
                fixed_bins = [100, 200, 500, 1000, 2000, 5000]
            elif col == 'rank_of_ban_current_is_biggest':
                fixed_bins = [1,2,3,4]

            n, bins, patches = plt.hist(data0[col].values, bins=fixed_bins, color='red', label='zero', alpha=0.5, normed=True)
            n, bins, patches = plt.hist(data1[col].values, bins=fixed_bins, color='blue', label='one', alpha=0.5, normed=True)

            plt.legend(loc='upper right')
            plt.xlabel(col)
            plt.ylabel('trouble_code_desc_n')
            # plt.title('About as simple as it gets, folks')
            plt.grid(True)
            plt.savefig('figures/'+key+'_'+col+'.png')
            # plt.show()


def print_detailed_outcomes(testdf, ypred, outputfilename=None):
    print ('shape: ', ypred.shape)
    testdf['ypred'] = ypred.reshape((len(ypred), 1))
    testdf['prediction'] = testdf['ypred'].map({0: 'NTF', 1:'TF'})

    if outputfilename is None:
        outputfilename = 'Y.csv'
    testdf.to_csv(outputfilename, sep=',')

    models = {}
    models['5031'] = testdf[testdf['first_rgmodel_5031NV-030'] == 1]
    models['5268'] = testdf[testdf['first_rgmodel_5268AC'] == 1]
    models['589'] = testdf[testdf['first_rgmodel_NVG589'] == 1]
    models['599'] = testdf[testdf['first_rgmodel_NVG599'] == 1]

    # models['5031'] = testdf[testdf['first_rgmodel'] == '5031NV-030']
    # models['5268'] = testdf[testdf['first_rgmodel'] == '5268AC']
    # models['589'] = testdf[testdf['first_rgmodel'] == 'NVG589']
    # models['599'] = testdf[testdf['first_rgmodel'] == 'NVG599']


    for key, val in models.iteritems():
        conf_mat = confusion_matrix(val['trouble_code_desc_n'], val['ypred'])
        recall = metrics.recall_score(val['trouble_code_desc_n'], val['ypred'])
        print ('model: ', key)
        print ('recall: ', recall )
        print ('counts: ' , sum(sum(conf_mat)))
        print conf_mat
        print ('--------')



#### read whole data
sdf = pd.read_table('/Users/Mz/Downloads/att_files/create_table/ml_mz.csv', sep=',')

print sdf.ix[:10,:]

date_col = 'swap_dt'
sdf_dict = prepare_data(sdf, date_col)

# create_hist(sdf_dict)

for key, sdf in sdf_dict.iteritems():
    print 'rg_model: ' , key
    # print sdf.shape
    # print sdf.columns
    # print sdf.ix[100:105,:]


    #### split data to train & test
    [train, test] = split_train_test(sdf, date_col)


    #### split train & test to x and y
    xtrain, ytrain, ratio = split_x_y(train)
    xtest, ytest, _chert_  = split_x_y(test)

    # print 'intervals: '
    # print min(xtrain[swap_datetime_col])
    # print max(xtrain[swap_datetime_col])
    # print min(xtest[swap_datetime_col])
    # print max(xtest[swap_datetime_col])




    print 'shapes:'
    print xtrain.shape
    # print ytrain.shape
    print xtest.shape
    # print ytest.shape

    print xtrain.columns

    # print xtrain.ix[:10,:].values



    #### Create linear regression model
    print ( 'train_columns: ', xtrain.columns)
    print ('train: ', xtrain.ix[:20,:])
    ypred , ypred_proba = simple_learning_model(xtrain.values, ytrain.values, xtest.values, ytest.values, ratio, key)
    resultdf = print_detailed_outcomes(test, ypred)

    #### testing new method from here:
    ypred, ypred_proba = simple_learning_model(xtrain.values, ytrain.values, xtrain.values, ytrain.values, ratio , key)
    resultdf = print_detailed_outcomes(train, ypred)
    print ' =================================================================='

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



