import pandas as pd
import numpy as np
import Util as U
import pickle


def prepare_data(data, separated_models=False, date_col='dt', model_col='model', dummy_encoding=False):


    data[date_col] =  pd.to_datetime(data[date_col], format='%Y-%m-%d %H:%M:%S.%f').dt.date

    #### convert time_of_day to nominal values
    data = U.to_nominal(data,col_name='time_of_day', nominal_col_name='time_of_day')

    #### filter rg_models
    model_values = ['NVG599', 'NVG589', '5268AC', '5031NV-030']
    data = data[data[model_col].isin(model_values)]

    U.report(data, 'nominalized labels')



    # separated_models = True
    if separated_models:
        # #### convert category to nominal for rg_model
        # data = U.to_nominal(data, model_col)
        #
        # #### split data based on rg_model
        # data_dict = split_base_on_column(data, model_col)
        #
        rgmodels_list = []

    else:
        rgmodels_list = data[model_col].unique().tolist()

        #### dummy encoding of rg_models
        dummy_data = pd.get_dummies(data,columns= [model_col], drop_first=dummy_encoding, prefix='', prefix_sep='')
        data_dict = {'all_model': dummy_data}
        rgmodels_list = [i  for i in rgmodels_list if i in dummy_data.columns]

    rgmodels_list = rgmodels_list
    print ('rgmodels_list: ', rgmodels_list)
    return data_dict, rgmodels_list


def select_columns(data, rgmodels_list= []):
    xcols = rgmodels_list + [ 'duration' , 'age_of_device' , 'use_factor' , 'rg_days_on_ban'  , 'calendar_day' , 'time_of_day']
    xdf = data[xcols]
    return xdf


def predict_from_proba(proba, threshold):
    pred = np.array([ 0 if proba[i,0] > threshold else 1 for i in range(proba.shape[0])])
    return pred;


def predict(x, day='', month='', pickle_filename=None):

    clf, method, ratio, rg_model= pickle.load(open(pickle_filename, 'rb'))
    if method == 'lr':
        ypred_proba = clf.predict_proba(x)
        ypred = predict_from_proba(ypred_proba, ratio)
    else:
        ypred_proba = None
        ypred = clf.predict(x)

    print (ypred.shape)
    print (ypred_proba.shape)

    return ypred, ypred_proba


def prepare_output(data, prediction, prediction_proba, date, prediction_col = 'prediction' , rood_dir = 'outputs'):

    data['passing_test_proba'] = prediction_proba[:,0] .round(decimals=2)
    data[prediction_col] = prediction
    data[prediction_col] = data[prediction_col].apply(lambda x: 'PASS' if x == 0 else 'FAIL')

    data.to_csv(rood_dir+'/predictions/'+date+'.csv', sep=',', index=False)

pickle_filename = 'outputs/pickles/lr_4.pickle'
test_date ={'year':2018, 'month':4, 'day':1}

data = pd.read_table('/Users/Mz/Downloads/att_files/create_table/tf_dispatch_20180401.csv', sep='\t')
date_col, model_col ='swap_date' ,  'model'


sdf_dict, rgmodels_list = prepare_data(data,  date_col=date_col, model_col=model_col)

for key, sdf in sdf_dict.iteritems():

    #### select columns needed for prediction
    rgmodels_list = ['NVG589', '5268AC', '5031NV-030', 'NVG599']
    xcols = rgmodels_list + [ 'duration' , 'age_of_device' , 'use_factor' , 'rg_days_on_ban'  , 'calendar_day' , 'time_of_day']
    xdf = sdf[xcols]

    #### predict label for input data
    ypred, ypred_proba = predict(xdf.values,  day=test_date['day'], month=test_date['month'], pickle_filename=pickle_filename)

    #### add prediction and probability to the input data
    prediction_col = 'prediction'
    prediction_proba_col = 'passing_test_probability'
    sdf[prediction_proba_col] = ypred_proba[:,0].round(decimals=2)
    sdf[prediction_col] = ypred

    #### save output file, containing inputs and predictions as csv
    output_filename='outputs/predictions_'+key+'_20180401.csv'
    sdf.to_csv(output_filename, index=False)

