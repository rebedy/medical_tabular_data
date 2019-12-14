# import matplotlib
# matplotlib.use('Agg')
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder  # https://brunch.co.kr/@sokoban/8
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn import preprocessing
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout
from sklearn.metrics import roc_auc_score
import time
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping, ModelCheckpoint
import datetime
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from itertools import cycle
from scipy import interp

def call_data(path):
    data = pd.read_csv(path, sep='delimiter', delimiter=',', encoding='euc-kr')
    
    data = data.rename(columns=lambda x: x.strip('\n'))
    data = data.drop(['ID'], axis=
                     1) 
    ##### Let's fill the N/A cells
    data = data.fillna('0') 
    key = data.keys()

    REGI_DATE = data['REGI_DATE']
    sex = data['SEX']
    F_KTAS = data['F_KTAS']
    ER_METHOD = data['ER_METHOD']
    age_section = data['Age_section']
    cc = data['CC']
    sbp = data['SBP']
    dbp = data['DBP']
    outcome = data['outcome']
    
    X_data = data[['F_KTAS','ER_METHOD','SEX','Age_section','CC','SBP','DBP','PR','RR','TEMP','SPO2','AVPU','ER_ROUTE']]
    Y_data = data['outcome']

    return X_data, Y_data


def ER_ROUTE_one_hot(one_hot_targets):
    label_encoder = LabelEncoder()
    
    ##### Encoding
    s_data = label_encoder.fit_transform(one_hot_targets)
    ER_ROUTE_one_hot = pd.get_dummies(s_data).values
    
    return ER_ROUTE_one_hot


def SEX_one_hot(one_hot_targets):
    label_encoder = LabelEncoder()
    
    ##### Encoding
    s_data = label_encoder.fit_transform(one_hot_targets)
    SEX_one_hot = pd.get_dummies(s_data).values

    return SEX_one_hot


def CC_one_hot(one_hot_targets):
    label_encoder = LabelEncoder()

    ##### Encoding
    s_data = label_encoder.fit_transform(one_hot_targets)
    CC_one_hot = pd.get_dummies(s_data).values

    return CC_one_hot


def AVPO_one_hot(one_hot_targets):
    label_encoder = LabelEncoder()

    ##### Encoding
    s_data = label_encoder.fit_transform(one_hot_targets)
    AVPO_one_hot = pd.get_dummies(s_data).values

    return AVPO_one_hot


""" ER method 4 class """
def ER_METHOD_4class(one_hot_targets):
    label_encoder = LabelEncoder()

    ##### Encoding
    eh_method_s_data = label_encoder.fit_transform(one_hot_targets)
    ER_METHOD_one_hot = pd.get_dummies(eh_method_s_data).values

    return ER_METHOD_one_hot

""" F_KTAS """
def fourclass_one_hot(one_hot_targets):
    one_hot_col, encoded_list = list(one_hot_targets), []
    for col in one_hot_col:
        one_hot_target = np.unique(one_hot_targets[col], return_inverse=1)[1] 
        one_hot_encoding = np.zeros((one_hot_target.shape[0], int((one_hot_target.max() + 1)/2)), dtype=int)  
        print("F_KTAS before one hot encoding this must be 3",one_hot_target[6])
        
        loop_num = 0
        for i in one_hot_target:
            if i == 0:
                one_hot_encoding[loop_num] = [0, 0]
            if i == 1:
                one_hot_encoding[loop_num] = [0, 1]
            if i == 2:
                one_hot_encoding[loop_num] = [1, 0]
            if i == 3:
                one_hot_encoding[loop_num] = [1, 1]
            loop_num = loop_num + 1
            
        ##### To change 'space' noticable.
        col = col.replace(" ", "_")
        encoded = np.array(one_hot_encoding)
        print("F_KTAS after one hot encoding this must be 3 and should be matched to [1, 1] ->", encoded[6])
        
        ##### Assigning variables in the loop.
        vars()[col] = pd.DataFrame(encoded)
        encoded_list.append(vars()[col])
        
    encoded_df = pd.concat(encoded_list, axis=1)

    return encoded_df


""" Age_section """
def ageclass_one_hot(one_hot_targets):
    one_hot_col = list(one_hot_targets)
    encoded_list = []
    for col in one_hot_col:
        one_hot_target = np.unique(one_hot_targets[col], return_inverse=1)[1]
        print("this must be 9",one_hot_target[37573])
        print("this must be 1",one_hot_target[11680])
        one_hot_encoding = np.zeros((one_hot_target.shape[0], 4), dtype=int)

        loop_num = 0
        for i in one_hot_target:
            if i == 0:
                one_hot_encoding[loop_num] = [0, 0, 0, 0]
            elif i == 1:
                one_hot_encoding[loop_num] = [1, 0, 0, 0]
            elif i == 2:
                one_hot_encoding[loop_num] = [0, 1, 0, 0]
            elif i == 3:
                one_hot_encoding[loop_num] = [0, 0, 1, 0]
            elif i == 4:
                one_hot_encoding[loop_num] = [0, 0, 0, 1]
            elif i == 5:
                one_hot_encoding[loop_num] = [1, 1, 0, 0]
            elif i == 6:
                one_hot_encoding[loop_num] = [1, 0, 1, 0]
            elif i == 7:
                one_hot_encoding[loop_num] = [1, 0, 0, 1]
            elif i == 8:
                one_hot_encoding[loop_num] = [1, 1, 1, 0]
            elif i == 9:
                one_hot_encoding[loop_num] = [1, 1, 0, 1]

            loop_num = loop_num + 1
            
        col = col.replace(" ", "_")

        encoded = np.array(one_hot_encoding)
        print("after age one hot this must be 9 and should be matched to [1, 1, 0, 1] ->", encoded[37573])
        print("after age one hot this must be 1 and should be matched to [1, 0, 0, 0] ->", encoded[11680])
        
        vars()[col] = pd.DataFrame(encoded)  # for loop 안에서 변수 생성
        encoded_list.append(vars()[col])

    encoded_df = pd.concat(encoded_list, axis=1)

    return encoded_df

def four_one_hot(one_hot_targets):
    one_hot_target = np.unique(one_hot_targets, return_inverse=1)[1]
    one_hot_encoding = np.zeros((one_hot_target.shape[0], one_hot_target.max()), dtype=int)
    loop_num = 0

    for i in one_hot_target:
        """
        ##### code 수정하기 
        예) 1. value가 1이면(for문 없이 value 대치 가능한지 확인) 
                -> 1이면 one_hot_encoding에서 1번째 값이 변경되도록 하기
                [0, 0, 0] -> [1, 0, 0]
        """
        if i == 5 or i == 0:
            one_hot_encoding[loop_num] = [0, 0, 0, 0]
        elif i == 1:
            one_hot_encoding[loop_num] = [0, 0, 0, 1]
        elif i == 2:
            one_hot_encoding[loop_num] = [0, 0, 1, 0]
        elif i == 3:
            one_hot_encoding[loop_num] = [0, 1, 0, 0]
        elif i == 4:
            one_hot_encoding[loop_num] = [1, 0, 0, 0]
        loop_num = loop_num + 1

    encoded = np.array(one_hot_encoding)
    encoded_df = pd.DataFrame(encoded)

    return encoded_df


def Y_one_hot_encoding(onehot_target):
    label_encoder = LabelEncoder()
    
    ##### Encoding
    Y_data = label_encoder.fit_transform(onehot_target)
    Y_data_one_hot = pd.get_dummies(Y_data).values

    return Y_data_one_hot


def pause():
    input("Please press keyboard ...")


def data_scaling(X_train):
    '''
    Scaling data in `MinMax Scaler`.
    '''
    from sklearn.preprocessing import MinMaxScaler
    
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit_transform(X_train)
    X_train = scaler.fit_transform(X_train)
    
    return X_train

def smote(X_train, y_train):
    '''
    To apply SMOTE Model.
       * SMOTE(synthetic minority oversampling technique) 
    '''
    from imblearn.over_sampling import SMOTE

    ##### SMOTE Model
    sm = SMOTE(ratio='auto', kind='regular')

    ##### Duplicate train dataset
    X_train, y_train = sm.fit_sample(X_train, list(y_train))
    print('After OverSampling, the shape of train_X: {}'.format(X_train.shape))
    print('After OverSampling, the shape of train_y: {} \n'.format(X_train.shape))
    print("After OverSampling, counts of label '1': {}".format(sum(y_train == 1)))
    print("After OverSampling, counts of label '0': {}".format(sum(y_train == 0)))

    
##### MAIN #####

data_path = '/Users/smc_ai_mac/Desktop/KTAS/data/datas.csv'

X_data, Y_data = call_data(path=data_path)
"F_KTAS' 4 class,'ER_METHOD 자차, 도보, 경찰차등 대중교통, 구급','SEX 2 class', Age"

# three one hot
sex_onehot = SEX_one_hot(one_hot_targets=X_data[['SEX']])
er_route_onehot = ER_ROUTE_one_hot(one_hot_targets=X_data[['ER_ROUTE']])
cc_onehot = CC_one_hot(one_hot_targets=X_data[['CC']])
avpo_onehot = AVPO_one_hot(one_hot_targets=X_data[['AVPU']])
ktas_one_hot_df = X_data[['F_KTAS']].astype(int)
F_KTAS_onehot = fourclass_one_hot(one_hot_targets=ktas_one_hot_df)
er_method = ER_METHOD_4class(one_hot_targets=X_data[['ER_METHOD']])
Age_section = X_data[['Age_section']].astype(int)
Age_section_onehot = ageclass_one_hot(one_hot_targets=Age_section)

X_data_onehot = pd.concat([pd.DataFrame(sex_onehot), F_KTAS_onehot, pd.DataFrame(er_method),
                           Age_section_onehot, pd.DataFrame(er_route_onehot), 
                           pd.DataFrame(cc_onehot), pd.DataFrame(avpo_onehot)],axis=1)

permutation = 10
kfold = 5
auc_list, unit_result, opti_result, batch_result, drop_result = [], [], [], [], []
unit_list = [100, 200, 300, 400]
opti_list = ['adam']
batch_list = [15]
drop_list = [0.5, 0.7]
best_auc_sum = -10000
best_auc_result, best_unit, best_opti, best_batch, best_drop = 0, 0, 0, 0, 0

now = datetime.datetime.now()
start_time = now.strftime('%H:%M:%S')
print("   Grid Search Starts @ : ", start_time)
print(" ------------------------------------ ")

grid_search = []
for unit in unit_list:
    for opti in opti_list:
        for batch in batch_list:
            for drop in drop_list:
                
                fold_num, fold_auc_result = 0, []
                cv = StratifiedKFold(n_splits=kfold, shuffle=True, random_state=42)
                                     
                for train_index, test_index in cv.split(X_data_onehot, Y_data):
                    fold_num = fold_num + 1
                    print("fold : ", fold_num)
                    
                    X_train, X_test = X_data_onehot.T[train_index].T, X_data_onehot.T[test_index].T
                    Y_train, Y_test = Y_data.T[train_index].T, Y_data.T[test_index].T
                    Y_train = Y_one_hot_encoding(onehot_target=Y_train)
                    Y_test = Y_one_hot_encoding(onehot_target=Y_test)
                                     
                    ##### Normalization
                    X_standardScaler = preprocessing.StandardScaler()
                    X_tr_nor = X_standardScaler.fit_transform(X_train)
                    X_te_nor = X_standardScaler.fit_transform(X_test)
                    print(X_tr_nor.shape)

                    ##### Model Definition
                    model = Sequential()
                    model.add(Dense(units=unit, activation='relu', input_dim=X_tr_nor.shape[1]))  # input layer
                    model.add(Dropout(drop))
                    model.add(Dense(units=unit, activation='relu'))  # first hidden layer
                    model.add(Dropout(drop))
                                    ##### units == node
                    model.add(Dense(units=Y_train.shape[1], activation='softmax'))  # output layer
                    
                    ##### Compile and Fit and Predic
                    model.compile(optimizer=opti, loss='categorical_crossentropy', metrics=['accuracy'])
                    model.fit(X_tr_nor, Y_train, epochs=5, batch_size=batch, verbose=2)
                    y_pred = model.predict(X_te_nor)

                    ##### Validation
                    auc_score = roc_auc_score(Y_test, y_pred, multi_class='ovr') ##### one vs rest
                    print("auc_score :", auc_score)
                    n_classes = Y_train.shape[1]
                    print("n_classes :", n_classes)
                    
                    ##### Compute ROC curve and ROC area for each class
                    fpr = dict()
                    tpr = dict()
                    roc_auc = dict()
                    for i in range(n_classes):
                        fpr[i], tpr[i], _ = roc_curve(Y_test[:, i], y_pred[:, i])
                        roc_auc[i] = auc(fpr[i], tpr[i])

                    ##### Compute micro-average ROC curve and ROC area
                    fpr["micro"], tpr["micro"], _ = roc_curve(Y_test.ravel(), y_pred.ravel())
                    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

                    ##### First aggregate all false positive rates
                    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

                    ##### Then interpolate all ROC curves at this points
                    mean_tpr = np.zeros_like(all_fpr)
                    for i in range(n_classes):
                        mean_tpr += interp(all_fpr, fpr[i], tpr[i])

                    ##### Finally average it and compute AUC
                    mean_tpr /= n_classes
                    
                    fpr["macro"] = all_fpr
                    tpr["macro"] = mean_tpr
                    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

                    ##### Plot all ROC curves
                    plt.figure()
                    lw = 2
                    plt.plot(fpr["micro"], tpr["micro"],
                             label='micro-average ROC curve (area = {0:0.2f})'.format(roc_auc["micro"]),
                             color='deeppink', linestyle=':', linewidth=4)
                    plt.plot(fpr["macro"], tpr["macro"],
                             label='macro-average ROC curve (area = {0:0.2f})'.format(roc_auc["macro"]),
                             color='navy', linestyle=':', linewidth=4)
                    colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
                    for i, color in zip(range(n_classes), colors):
                        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                                 label='ROC curve of class {0} (area = {1:0.2f})'.format(i, roc_auc[i]))
                    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
                    plt.xlim([0.0, 1.0])
                    plt.ylim([0.0, 1.05])
                    plt.xlabel('False Positive Rate')
                    plt.ylabel('True Positive Rate')
                    plt.title('Some extension of Receiver operating characteristic to multi-class')
                    plt.legend(loc="lower right")
                    plt.show()

                    fold_auc_result.append(auc)
                    fold_result = sum(fold_auc_result) / kfold
                    parameter = [unit, opti, batch, drop]
                    grid_search.append(parameter)
                    print("fold_result", fold_result)
                
                print("grid_search",grid_search)
                print("fold_auc_result",fold_auc_result)

                for itr, value in enumerate(fold_auc_result):
                    if max(fold_auc_result) == value:
                        # fold_mean_auc_value = fold_result
                        parameter = [unit, opti, batch, drop]
                        best_unit = unit
                        best_opti = opti
                        best_batch = batch
                        best_drop = drop
                        param_df = pd.DataFrame(parameter).T

                        print("best result",value)
                        print("parameter",parameter)

                        end_time = now.strftime('%H:%M:%S')
                        save_result1 = pd.DataFrame({'auc': best_auc_result,
                                                     'start_time': start_time,
                                                     'end_time': end_time})
                        save_param1 = param_df

                        try:
                            # Create target Directory
                            os.mkdir('/Users/smc_ai_mac/Desktop/KTAS/' + str(unit)+'node_model')
                            # print("Directory ", dirName, " Created ")
                        except FileExistsError:
                            pass
                        log_dir = '/Users/smc_ai_mac/Desktop/KTAS/' + str(unit)+'model/'  # local 경로
                        os.chdir(log_dir)  # 경로 변경
                        save_result1.to_csv('Std_zero_soft_layer1_auc(191202).csv')
                        save_param1.to_csv('Std_zero_soft_layer1_param(191202).csv')

                    else:
                        pass

print("Finished!!!!!!!!!@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
pause()

per_auc_avg_result = []
fold_num = 0
epoch = 200
best_auc_sum = 0

for per in range(permutation):
    fold_auc_result = []
    cv = StratifiedKFold(n_splits=kfold, shuffle=True, random_state=42 + per)
    
    for train_index, test_index in cv.split(X_data_onehot, Y_data):
        fold_num = fold_num + 1
        print("fold : ", fold_num)
        X_train, X_test = X_data_onehot.T[train_index].T, X_data_onehot.T[test_index].T
        Y_train, Y_test = Y_data.T[train_index].T, Y_data.T[test_index].T
        Y_train = Y_one_hot_encoding(onehot_target=Y_train)
        Y_test = Y_one_hot_encoding(onehot_target=Y_test)

        ##### Normalization
        X_standardScaler = preprocessing.StandardScaler()
                                      # .MinMaxScaler(range=(0, 1))
        X_tr_nor = X_standardScaler.fit_transform(X_train)
        X_te_nor = X_standardScaler.fit_transform(X_test)
        print(X_tr_nor.shape)

        model = Sequential()
        model.add(Dense(units=unit, activation='relu', input_dim=X_tr_nor.shape[1]))  # input layer
        model.add(Dropout(drop))
        model.add(Dense(units=unit, activation='relu'))  # first hidden layer
        model.add(Dropout(drop))                # 'sigmoid'
        model.add(Dense(units=Y_train.shape[1], activation='softmax'))  # output layer
        
        # unit이 1이면,     activation = 'sigmoid' --> loss = 'binary_crossentropy'
        model.compile(optimizer=opti, loss='categorical_crossentropy', metrics=['accuracy', 'mse'])

        ##### TODO | DY : Check here please 
        ##model.fit(X_tr_nor, Y_train, epochs=batch, batch_size=batch)

        model_save_path = '/home/kym/Otorhinolaryngology/result/Model/'  # GPU 경로
        model_name = '%s_StdZeroSoftL1_model.h5' % str(fold_num)
        model_save = os.path.join(model_save_path, model_name)
        checkpoint = [ModelCheckpoint(filepath=model_save, save_best_only=True, monitor='val_loss')]

        history = model.fit(X_tr_nor, Y_train, epochs=epoch, batch_size=best_batch,
                            validation_data=(X_te_nor, Y_test),
                            callbacks=checkpoint, verbose=1)

        ###### Save weight
        model_save_path = '/home/kym/Otorhinolaryngology/result/Model/'  # GPU 경로
        weight_name = '%s_StdZeroSoftL1_model_weight' % str(fold_num)
        weight_save = os.path.join(model_save_path, weight_name)
        model.save_weights(weight_save)

        saved_model = load_model(model_save)
        saved_weight = saved_model.load_weights(weight_save)

        y_pred = saved_model.predict(X_te_nor)

        ###### Validation
        auc = roc_auc_score(Y_test, y_pred)
        print("auc :", auc)
        fold_auc_result.append(auc)

    fold_result = sum(fold_auc_result) / kfold
    per_auc_avg_result.append(fold_result)
    
per_result = sum(per_auc_avg_result)

if (best_auc_sum < per_result):
    best_auc_sum = per_result
    best_auc_result = per_auc_avg_result
    parameter = [unit, opti, batch, drop]
    param_df = pd.DataFrame(parameter).T
    print("best result")
    print(best_auc_sum, best_auc_sum / permutation)
    print(best_auc_result)
    print(parameter)
    
    save_result1 = pd.DataFrame({'auc': best_auc_result,'start_time': start_time, 'end_time': end_time})
    save_param1 = param_df

    log_dir = '/home/kym/Otorhinolaryngology/result/NeuralNet/10permutation/'  # GPU 경로
    os.chdir(log_dir)  # 경로 변경
    save_result1.to_csv('Std_zero_soft_layer1_auc_10per(191202).csv')
    save_param1.to_csv('Std_zero_soft_layer1_param_10per(191202).csv')
    
    end_time = now.strftime('%H:%M:%S')
    print(" ------------------------ ")
    print("   Ends @ ", end_time)
