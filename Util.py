import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
import datetime as dt
from sklearn.utils import resample

class transformation:

    tr_mean = 0
    tr_var = 1

    def transform(self,data, column):
        self.tr_mean = data[column].mean()
        self.tr_var = data[column].var()

        data[column] = data[column].map(lambda x: (x - self.tr_mean)/ self.tr_var )

    def transform_back(self, values):
        values = (values * self.tr_var ) + self.tr_mean
        return values

def report ( data, text = '' ):
    print ( '{0} {1} ---- {2} {3}'.format('<'*10 , text, type(data), '>'*10))
    if isinstance(data, dict):
        report ( data[data.keys()[0]], data.keys()[0])
    else:

        data.reset_index(drop=True, inplace=True)
        print ( 'shape: ', data.shape)
        print ( 'columns: ', data.columns)
        print ( data.ix[:min(10,data.shape[0]),:])
    print ( '='*10 )


def to_nominal(data, col_name, values=[], nominal_col_name=None):
    if len(values) == 0:
        values = data[col_name].unique().tolist()
    if nominal_col_name is None:
        nominal_col_name = col_name+'_n'
    data = data[data[col_name].isin(values)]
    data[nominal_col_name] = data[col_name].astype("category", ordered=True, categories=values).cat.codes
    data.reset_index(drop=True, inplace=True)
    return data

def upsample(data, label_col):
    # Separate majority and minority classes
    df_majority = data[data[label_col]==0]
    df_minority = data[data[label_col]==1]

    n_samples = df_majority.shape[0]
    print ('n_samples: ', n_samples)

    # Upsample minority class
    df_minority_upsampled = resample(df_minority,
                                     replace=True,     # sample with replacement
                                     n_samples=n_samples,    # to match majority class
                                     random_state=123) # reproducible results

    print ('df_minority_upsampled: ', df_minority_upsampled.shape, ' , ' , df_majority.shape)
    # Combine majority class with upsampled minority class
    df_upsampled = pd.concat([df_majority, df_minority_upsampled])
    print ('df_upsampled: ', df_upsampled.shape)
    # Display new class counts
    print ('value_counts: ', df_upsampled[label_col].value_counts() )
    return df_upsampled

def stack( arr, arr_all=None, axis=0):
    #### axis=0 : x*n , k*n ==> (k+x)*n
    print ('stack_folds_preds...')
    if arr_all is None:
        arr_all = arr
    else:
        try:
            # if axis==0:
            arr_all = np.vstack((arr_all, arr)) if axis==0 else np.hstack((arr_all, arr))
            # else:
            #     pred_all = np.vstack((pred_all, pred_fold))
        except Exception as e:
            print ('error in stack: ( ', e ,' )')
            print ('arr.shape: ', arr.shape)
            print ('arr_all.shape: ' ,arr_all.shape)

    return arr_all


def outlier_detection(data, column = 'logerror' , thresh=1):
    print ('outlier_detection...')
    print (data.shape)
    # print type(data)
    # data = data[np.where(data[:,0]<0.9)]
    data = data[abs(data[column] ) < thresh]
    print (data.shape)
    return data


def into_classes(data):
    # print 'data: ' , data
    if abs(data)<=1:
        return 0
    elif data<-1:
        return -1
    else: return 1
    # mid = data[abs(data.logerror)<=1]
    # low = data[data.logerror< -1]
    # high = data[data.logerror> 1]
    # print low.shape, ' , ', mid.shape, ' , ', high.shape



class mean_est:
    def __init__(self,type='regression'):
        self.type = type
        self.mean = None

    def fit(self, X, y):
        print ('fit: ' , X.shape ,  '  , ' , y.shape)
        self.mean = np.mean(y)
        if self.type is 'classification':
            self.mean = np.sign(self.mean)

    def predict(self, X):
        print ('predict: ' , X.shape )
        return np.array([ self.mean for i in range(X.shape[0])])



# def evaluate(Ytrue, Ypred, type='regression',  pre = 'pre ', mea=None, va=None):
#     if not mea is None:
#         Ytrue = transform_back(Ytrue, mea, va)
#         Ypred = transform_back(Ypred, mea, va)
#
#     mae = mean_absolute_error(Ytrue,Ypred)
#     mse = mean_squared_error(Ytrue,Ypred)
#     with open("res.txt", "a") as myfile:
#
#         if type is 'regression':
#             myfile.write(pre + 'mae: ' + str(mae)+ ' , mse: ' + str(mse) + ' \n' )
#             print ('mae: ' , mae, ' , mse: ', mse)
#         elif type is 'classification2':
#             print ('accuracy: ' , (2-mae)/2)
#     return [mae , mse]

def evaluate(Ytrue, Ypred, type='regression',  pre = 'pre ', trnsfrm = None):
    print ('evaluate...')
    if not trnsfrm is None:
        Ytrue = trnsfrm.transform_back(Ytrue)
        Ypred = trnsfrm.transform_back(Ypred)

    # print ('before mae')
    mae = mean_absolute_error(Ytrue,Ypred)
    mse = mean_squared_error(Ytrue,Ypred)
    # print ('mae: '  , mae)
    with open("res.txt", "a") as myfile:
        if type is 'regression':
            # print ('type: ' , type)
            myfile.write(pre + 'mae: ' + str(mae)+ ' , mse: ' + str(mse) + ' \n' )
            print (pre, ' mae: ' , mae, ' , mse: ', mse)
        elif type is 'classification2':
            print ('accuracy: ' , (2-mae)/2)
    return [mae , mse]


def cats_to_int_1param(data):
        print ('cats_to_int...')
        cat_columns = data.select_dtypes(['category','object']).columns
        print ('cat_columns: ' , cat_columns)

        for col in cat_columns:
            print ('col:  ' , col)
            data[col] = pd.Categorical(data[col])
            data[col] = data[col].astype('category').cat.codes
        return data

def cats_to_int(data1, data2):
        print ('cats_to_int...')
        cat_columns = data1.select_dtypes(['category','object']).columns
        print ('cat_columns: ' , cat_columns)

        for col in cat_columns:
            if col == 'propertyzoningdesc':
                continue
            print ('col:  ' , col)
            l1 = list(set(data1[col].values))
            l2 = list(set(data2[col].values))
            # print 'l1, l2: ' , l1, ' , ', l2
            l = l1 + l2
            l = [ ll for ll in l if ll==ll]
            # print 'l: ' , l
            l = list(set(l))
            print 'len l: ' , len(l)
            # l = list(l1.update(l2))
            # print ('l: ' , l)
            data1[col] = [ l.index(c) if c in l else c for c in data1[col]  ]
            data2[col] = [ l.index(c) if c in l else c for c in data2[col]  ]

        return data1, data2




def add_date_features(df, drop_transactiondate=True):
    df["transaction_year"] = df["transactiondate"].dt.year
    df["transaction_month"] = (df["transactiondate"].dt.year - 2016)*12 + df["transactiondate"].dt.month
    # df["transaction_day"] = df["transactiondate"].dt.day
    df["transaction_quarter"] = (df["transactiondate"].dt.year - 2016)*4 +df["transactiondate"].dt.quarter
    if drop_transactiondate:
        df.drop(["transactiondate"], inplace=True, axis=1)
    return df

def get_submission_format(data):
    print ('get_submission_format ...')
    data = data[['ParcelId']]
    # print (data)
    # cols = ['ParcelId', '10/1/16', '11/1/16', '12/1/16', '10/1/17', '11/1/17', '12/1/17']
    cols = ['ParcelId', '2016-10-1', '2016-11-1', '2016-12-1', '2017-10-1', '2017-11-1', '2017-12-1']
    # pd.Timestamp('2016-09-30')
    for i in range(1 ,len(cols)):
        c = cols[i]
        data[c] = 0
    data.columns = cols
    print (data.columns)
    print (data.shape)
    submission_df = pd.melt(data, id_vars=["ParcelId"],var_name="transactiondate", value_name="logerror")
    print ('submission_df: ')
    submission_df['transactiondate'] = [pd.Timestamp(str(s)) for s in  submission_df['transactiondate']]
    # print (submission_df)
    print (submission_df.shape)
    return submission_df

def prepare_final_submission(submission_df, Ypred, type=0, output_filename='data/final_submission_outlierDetection.csv'):
    print ('prepare_final_submission ...')
    print ('submission_df.columns: ' , submission_df.columns, ' , ', submission_df.shape)

    print ('Ypred: ', Ypred)
    ##### prepare submission dataframe to look like the actual submission file (using pivot_table)
    submission_df['logerror'] = Ypred

    submission_df = submission_df[['logerror']]

    print ('submission_df.columns: ' , submission_df.columns)

    # if ('Date' in submission_df.columns):
    # if type == 0:
    submission_df.reset_index(inplace=True)
    print (submission_df)


    submission_df = submission_df.pivot_table(values='logerror', index='ParcelId', columns='transactiondate')

    submission_df.reset_index(inplace=True)

    submission_df.columns = ['ParcelId' , '201610' , '201710', '201611', '201711', '201612', '201712']

    submission_df = submission_df[['ParcelId' , '201610' ,  '201611', '201612', '201710','201711', '201712' ]]

    submission_df.set_index('ParcelId', inplace=True)


    # else:
    #     cols = ['201610' , '201611', '201612', '201710', '201711', '201712']
    #     for i in range(len(cols)):
    #         c = cols[i]
    #         submission_df[c] = submission_df['logerror']
    #     submission_df = submission_df[cols]


    print ('final_submission_df.shape: ' , submission_df.shape)
    print ('final_submission_df.columns: ' , submission_df.columns)
    print (submission_df)
    # final_submission_name = 'data/final_submission_outlierDetection.csv'

    submission_df.to_csv(output_filename)

