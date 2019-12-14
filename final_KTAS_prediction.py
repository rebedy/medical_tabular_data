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
    data = data.rename(columns=lambda x: x.strip(
        '\n'))  # column 이름에 \n 출력 되는 것 삭제 #https://stackoverflow.com/questions/21606987/how-can-i-strip-the-whitespace-from-pandas-dataframe-headers
    data = data.drop(['ID'], axis=1)  # 필요 없는 정보 삭제
    # print("data",data)

    # data = data.iloc[:313, :]  # 313, 33

    # data = data.fillna(data.median(skipna=True))  # NaN값에 중앙값 채우기
    data = data.fillna('0')  # NaN값에 중앙값 채우기

    key = data.keys()
    # print("key",key)
    # pause()

    # val = data.columns()
    REGI_DATE = data['REGI_DATE']
    sex = data['SEX']
    F_KTAS = data['F_KTAS']
    ER_METHOD = data['ER_METHOD']
    # BIRTH = data['BIRTH']
    outcome = data['outcome']
    age_section = data['Age_section']
    cc = data['CC']
    sbp = data['SBP']
    dbp = data['DBP']

    # print("ER_METHOD",ER_METHOD)
    # print("outcome",outcome)


    X_data = data[['F_KTAS','ER_METHOD','SEX','Age_section','CC','SBP','DBP','PR','RR','TEMP','SPO2','AVPU','ER_ROUTE']]
    Y_data = data['outcome']

    #    print(Y_data['Fistula'].value_counts())    #value counting

    return X_data, Y_data




def ER_ROUTE_one_hot(one_hot_targets):

    label_encoder = LabelEncoder()

    # Y data 인코딩
    s_data = label_encoder.fit_transform(one_hot_targets)
    ER_ROUTE_one_hot = pd.get_dummies(s_data).values

    # print("SEX_one_hot",SEX_one_hot)

    # one_hot_target = np.unique(one_hot_targets, return_inverse=1)[1]  # one_hot 하고자 하는 column
    # one_hot_encoding = np.zeros((one_hot_target.shape[0], one_hot_target.max()), dtype=int)  # one_hot 대상만큼 zero,
    # loop_num = 0
    #
    # # for j in range(one_hot_target.shape[0]):
    # for i in one_hot_target:
    #     # code 수정하기 예) 1.value가 1이면(for문 없이 value 대치 가능한지 확인) -> 1이면 one_hot_encoding에서 1번째 값이 변경되도록 하기 [0, 0, 0] -> [1, 0, 0]
    #     if i == 1:
    #         one_hot_encoding[loop_num] = [0, 1]
    #     if i == 2:
    #         one_hot_encoding[loop_num] = [1, 0]
    #     loop_num = loop_num + 1
    #
    # encoded = np.array(one_hot_encoding)
    # encoded_df = pd.DataFrame(encoded)

    return ER_ROUTE_one_hot

def SEX_one_hot(one_hot_targets):

    label_encoder = LabelEncoder()

    # Y data 인코딩
    s_data = label_encoder.fit_transform(one_hot_targets)
    SEX_one_hot = pd.get_dummies(s_data).values

    # print("SEX_one_hot",SEX_one_hot)

    # one_hot_target = np.unique(one_hot_targets, return_inverse=1)[1]  # one_hot 하고자 하는 column
    # one_hot_encoding = np.zeros((one_hot_target.shape[0], one_hot_target.max()), dtype=int)  # one_hot 대상만큼 zero,
    # loop_num = 0
    #
    # # for j in range(one_hot_target.shape[0]):
    # for i in one_hot_target:
    #     # code 수정하기 예) 1.value가 1이면(for문 없이 value 대치 가능한지 확인) -> 1이면 one_hot_encoding에서 1번째 값이 변경되도록 하기 [0, 0, 0] -> [1, 0, 0]
    #     if i == 1:
    #         one_hot_encoding[loop_num] = [0, 1]
    #     if i == 2:
    #         one_hot_encoding[loop_num] = [1, 0]
    #     loop_num = loop_num + 1
    #
    # encoded = np.array(one_hot_encoding)
    # encoded_df = pd.DataFrame(encoded)

    return SEX_one_hot

def CC_one_hot(one_hot_targets):
    print("one_hot_targets",one_hot_targets)

    label_encoder = LabelEncoder()

    # Y data 인코딩
    s_data = label_encoder.fit_transform(one_hot_targets)
    CC_one_hot = pd.get_dummies(s_data).values

    # print("SEX_one_hot",SEX_one_hot)

    # one_hot_target = np.unique(one_hot_targets, return_inverse=1)[1]  # one_hot 하고자 하는 column
    # one_hot_encoding = np.zeros((one_hot_target.shape[0], one_hot_target.max()), dtype=int)  # one_hot 대상만큼 zero,
    # loop_num = 0
    #
    # # for j in range(one_hot_target.shape[0]):
    # for i in one_hot_target:
    #     # code 수정하기 예) 1.value가 1이면(for문 없이 value 대치 가능한지 확인) -> 1이면 one_hot_encoding에서 1번째 값이 변경되도록 하기 [0, 0, 0] -> [1, 0, 0]
    #     if i == 1:
    #         one_hot_encoding[loop_num] = [0, 1]
    #     if i == 2:
    #         one_hot_encoding[loop_num] = [1, 0]
    #     loop_num = loop_num + 1
    #
    # encoded = np.array(one_hot_encoding)
    # encoded_df = pd.DataFrame(encoded)

    return CC_one_hot

def AVPO_one_hot(one_hot_targets):

    label_encoder = LabelEncoder()

    # Y data 인코딩
    s_data = label_encoder.fit_transform(one_hot_targets)
    AVPO_one_hot = pd.get_dummies(s_data).values

    # print("SEX_one_hot",SEX_one_hot)

    # one_hot_target = np.unique(one_hot_targets, return_inverse=1)[1]  # one_hot 하고자 하는 column
    # one_hot_encoding = np.zeros((one_hot_target.shape[0], one_hot_target.max()), dtype=int)  # one_hot 대상만큼 zero,
    # loop_num = 0
    #
    # # for j in range(one_hot_target.shape[0]):
    # for i in one_hot_target:
    #     # code 수정하기 예) 1.value가 1이면(for문 없이 value 대치 가능한지 확인) -> 1이면 one_hot_encoding에서 1번째 값이 변경되도록 하기 [0, 0, 0] -> [1, 0, 0]
    #     if i == 1:
    #         one_hot_encoding[loop_num] = [0, 1]
    #     if i == 2:
    #         one_hot_encoding[loop_num] = [1, 0]
    #     loop_num = loop_num + 1
    #
    # encoded = np.array(one_hot_encoding)
    # encoded_df = pd.DataFrame(encoded)

    return AVPO_one_hot

"ER method 4 class"
def ER_METHOD_4class(one_hot_targets):
    label_encoder = LabelEncoder()

    # Y data 인코딩
    eh_method_s_data = label_encoder.fit_transform(one_hot_targets)
    # print("method class",np.unique(eh_method_s_data))
    # pause()

    ER_METHOD_one_hot = pd.get_dummies(eh_method_s_data).values

    return ER_METHOD_one_hot

"F_KTAS"
def fourclass_one_hot(one_hot_targets):
    one_hot_col = list(one_hot_targets)
    encoded_list = []
    # print("one_hot_col",one_hot_col)
    # pause()

    for col in one_hot_col: # 전체 하나씩 까주는 거.
        one_hot_target = np.unique(one_hot_targets[col], return_inverse=1)[1]  # one_hot 하고자 하는 column
        one_hot_encoding = np.zeros((one_hot_target.shape[0], int((one_hot_target.max() + 1)/2)),
                                    dtype=int)  # one_hot 대상만큼 zero,
        # print("np.unique(one_hot_target)",np.where(one_hot_target == 3))
        print("F_KTAS before one hot encoding this must be 3",one_hot_target[6])

        loop_num = 0

        for i in one_hot_target:  # i==0이면 [0, 0] 상태이기 때문에 코딩 필요 X
            if i == 0:
                one_hot_encoding[loop_num] = [0, 0]
            if i == 1:
                one_hot_encoding[loop_num] = [0, 1]
            if i == 2:
                one_hot_encoding[loop_num] = [1, 0]
            if i == 3:
                one_hot_encoding[loop_num] = [1, 1]
            loop_num = loop_num + 1

        col = col.replace(" ", "_")  # 띄어쓰기 때문에 변수명 안 되는 대상 띄어쓰기 -> _로 변경

        encoded = np.array(one_hot_encoding)
        print("F_KTAS after one hot encoding this must be 3 and should be matched to [1, 1] ->", encoded[6])

        vars()[col] = pd.DataFrame(encoded)  # for loop 안에서 변수 생성

        encoded_list.append(vars()[col])

    encoded_df = pd.concat(encoded_list, axis=1)

    return encoded_df


"Age_section"
def ageclass_one_hot(one_hot_targets):
    one_hot_col = list(one_hot_targets)
    encoded_list = []

    for col in one_hot_col: # 전체 하나씩 까주는 거.
        one_hot_target = np.unique(one_hot_targets[col], return_inverse=1)[1]  # one_hot 하고자 하는 column
        print("this must be 9",one_hot_target[37573])
        print("this must be 1",one_hot_target[11680])
        # print("np.unique(one_hot_target)",np.where(one_hot_target == 1))

        one_hot_encoding = np.zeros((one_hot_target.shape[0], 4),
                                    dtype=int)  # one_hot 대상만큼 zero,
        # print("one_hot_target.shape[0]",one_hot_target.shape[0])

        loop_num = 0
        for i in one_hot_target:  # i==0이면 [0, 0] 상태이기 때문에 코딩 필요 X
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
            # print("one_hot_encoding",one_hot_encoding[loop_num])
            # pause()
        col = col.replace(" ", "_")  # 띄어쓰기 때문에 변수명 안 되는 대상 띄어쓰기 -> _로 변경

        encoded = np.array(one_hot_encoding)
        print("after age one hot this must be 9 and should be matched to [1, 1, 0, 1] ->", encoded[37573])
        print("after age one hot this must be 1 and should be matched to [1, 0, 0, 0] ->", encoded[11680])
        vars()[col] = pd.DataFrame(encoded)  # for loop 안에서 변수 생성

        encoded_list.append(vars()[col])

    encoded_df = pd.concat(encoded_list, axis=1)

    # print("encoded_df",encoded_df)

    return encoded_df

def four_one_hot(one_hot_targets):
    one_hot_target = np.unique(one_hot_targets, return_inverse=1)[1]  # one_hot 하고자 하는 column
    one_hot_encoding = np.zeros((one_hot_target.shape[0], one_hot_target.max()), dtype=int)  # one_hot 대상만큼 zero,
    loop_num = 0

    # for j in range(one_hot_target.shape[0]):
    for i in one_hot_target:
        # code 수정하기 예) 1.value가 1이면(for문 없이 value 대치 가능한지 확인) -> 1이면 one_hot_encoding에서 1번째 값이 변경되도록 하기 [0, 0, 0] -> [1, 0, 0]
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
    # encoder
    label_encoder = LabelEncoder()

    # Y data 인코딩
    Y_data = label_encoder.fit_transform(onehot_target)
    Y_data_one_hot = pd.get_dummies(Y_data).values

    return Y_data_one_hot

def pause():
    input("Please type input............")


def data_scaling(X_train):
    "data 스케일링을 하고자 할때, 민맥스 스칼라로 하는 방"
    from sklearn.preprocessing import MinMaxScaler

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit_transform(X_train)
    X_train = scaler.fit_transform(X_train)

def smote(X_train, y_train):
    from imblearn.over_sampling import SMOTE
    "SMOTE를 적용하기 위한 코드. " \
    "SMOTE알고리즘은 오버샘플링 기법 중 합성데이터를 생성하는 방식으로 가장 많이 사용되고 있는 모델이다." \
    "SMOTE(synthetic minority oversampling technique)란, " \
    "합성 소수 샘플링 기술로 다수 클래스를 샘플링하고 기존 소수 샘플을 보간하여 새로운 소수 인스턴스를 합성해낸다." \
    "일반적인 경우 성공적으로 작동하지만, 소수데이터들 사이를 보간하여 작동하기 때문에 모델링셋의 소수데이터들 사이의 특성만을 반영하고 " \
    "새로운 사례의 데이터 예측엔 취약할 수 있다." \
    "https://mkjjo.github.io/python/2019/01/04/smote_duplicate.html" \
    "Imbalnce : https://imbalanced-learn.readthedocs.io/en/stable/index.html"

    # 모델설정
    sm = SMOTE(ratio='auto', kind='regular')

    # train데이터를 넣어 복제함
    X_train, y_train = sm.fit_sample(X_train, list(y_train))

    print('After OverSampling, the shape of train_X: {}'.format(X_train.shape))
    print('After OverSampling, the shape of train_y: {} \n'.format(X_train.shape))

    print("After OverSampling, counts of label '1': {}".format(sum(y_train == 1)))
    print("After OverSampling, counts of label '0': {}".format(sum(y_train == 0)))



# https://stackoverflow.com/questions/43516456/one-hot-encode-a-binary-value-in-numpy
# def zero_one_hot(one_hot_target) :

# For test(not using zero_one_hot encoding)
# def get_dummies(one_hot_target, one_hot_col) :
#   for i in one_hot_col :
#       one_hot_target[i] = get_dummies (~ 수정하기)


data_path = '/Users/smc_ai_mac/Desktop/KTAS/data/datas.csv'

X_data, Y_data = call_data(path=data_path)

"F_KTAS' 4 class,'ER_METHOD 자차, 도보, 경찰차등 대중교통, 구급','SEX 2 class', Age"


# three one hot
sex_onehot = SEX_one_hot(one_hot_targets=X_data[['SEX']])
# print("two_one_hot_df.head()",sex_onehot)
er_route_onehot = ER_ROUTE_one_hot(one_hot_targets=X_data[['ER_ROUTE']])

cc_onehot = CC_one_hot(one_hot_targets=X_data[['CC']])
avpo_onehot = AVPO_one_hot(one_hot_targets=X_data[['AVPU']])


ktas_one_hot_df = X_data[['F_KTAS']].astype(int)
F_KTAS_onehot = fourclass_one_hot(one_hot_targets=ktas_one_hot_df)

er_method = ER_METHOD_4class(one_hot_targets=X_data[['ER_METHOD']])

Age_section = X_data[['Age_section']].astype(int)
Age_section_onehot = ageclass_one_hot(one_hot_targets=Age_section)

X_data_onehot = pd.concat([pd.DataFrame(sex_onehot),F_KTAS_onehot, pd.DataFrame(er_method), Age_section_onehot, pd.DataFrame(er_route_onehot), pd.DataFrame(cc_onehot), pd.DataFrame(avpo_onehot)],axis=1)  # 313, 38    #8 / 3 / 27

permutation = 10

kfold = 5
auc_list = []
unit_result = []
opti_result = []
batch_result = []
drop_result = []

# cross validation
# fold_best_auc = -100

best_auc_sum = -10000
best_auc_result = 0

#unit == node

# unit_list = [4, 8, 16, 32, 64]
unit_list = [100, 200, 300, 400]

# opti_list = ['adam', 'adagrad', 'nadam','rmsprop']
opti_list = ['adam']
batch_list = [15]
drop_list = [0.5, 0.7]

#####

"SMOT 방법으로 minority class를 조합해서 synsetic data를 만드는 것. " \
"1,3의 가중 평균을 갖고 만드는 방법으로 imbalance 해결 한다."

"순수도를 비교하는 방법. " \

"매크로와 마이크로는 멀티 클래스에서 " \
"임밸런스 심한것을 hard 0.3으로 cutoff 를 줘서 hard dicision(하드 디시젼)을 해서 imbalance에서 이걸 했다는 식으로 가자." \
"중증한 환자를 확진 할거냐? " \
"" \

"early stopping 넣을 것."
"튜닝, 임밸런스 해결, 노말과 넌 노말로 바이너리로 바꿀 때는 어캐 될까?에 대해 고민." \

best_unit = 0
best_opti = 0
best_batch = 0
best_drop = 0

now = datetime.datetime.now()
start_time = now.strftime('%H:%M:%S')

grid_search = []

for unit in unit_list:
    for opti in opti_list:
        for batch in batch_list:
            for drop in drop_list:

                fold_num = 0
                fold_auc_result = []

                cv = StratifiedKFold(n_splits=kfold, shuffle=True, random_state=42)

                for train_index, test_index in cv.split(X_data_onehot, Y_data):
                    fold_num = fold_num + 1
                    print("fold : ", fold_num)

                    X_train, X_test = X_data_onehot.T[train_index].T, X_data_onehot.T[test_index].T
                    Y_train, Y_test = Y_data.T[train_index].T, Y_data.T[test_index].T

                    Y_train = Y_one_hot_encoding(onehot_target=Y_train)
                    Y_test = Y_one_hot_encoding(onehot_target=Y_test)

                    # Normalization
                    X_standardScaler = preprocessing.StandardScaler()
                    # 평균이 0, 표준편차가 1이 되도록 변환. #https://datascienceschool.net/view-notebook/f43be7d6515b48c0beb909826993c856/
                    X_tr_nor = X_standardScaler.fit_transform(X_train)
                    X_te_nor = X_standardScaler.fit_transform(X_test)

                    print(X_tr_nor.shape)

                    model = Sequential()  # https://gist.github.com/NiharG15/cd8272c9639941cf8f481a7c4478d525
                    model.add(Dense(units=unit, activation='relu', input_dim=X_tr_nor.shape[1]))  # input layer
                    model.add(Dropout(drop))
                    model.add(Dense(units=unit, activation='relu'))  # first hidden layer
                    model.add(Dropout(drop))
                    # units == node
                    model.add(Dense(units=Y_train.shape[1], activation='softmax'))  # output layer
                    # softmax보다 sigmoid가 더 좋은 성능을 보임
                    model.compile(optimizer=opti, loss='categorical_crossentropy', metrics=['accuracy'])

                    # model.fit(X_tr_nor, Y_train, epochs=200, batch_size=30, verbose=2)
                    model.fit(X_tr_nor, Y_train, epochs=5, batch_size=batch, verbose=2)
                    y_pred = model.predict(X_te_nor)
                    "1 vs 나머지 식으로 binary 로 만들어서 계산하는것. 매크로는 단순히 평군 냇것. 마이크로 매크로 에버리ㄴ"

                    # model validation
                    auc_score = roc_auc_score(Y_test, y_pred, multi_class='ovr')
                    print("auc_score :", auc_score)
                    n_classes = Y_train.shape[1]
                    print("n_classes :", n_classes)
                    ""
                    # Compute ROC curve and ROC area for each class
                    fpr = dict()
                    tpr = dict()
                    roc_auc = dict()
                    for i in range(n_classes):
                        fpr[i], tpr[i], _ = roc_curve(Y_test[:, i], y_pred[:, i])
                        roc_auc[i] = auc(fpr[i], tpr[i])

                    # Compute micro-average ROC curve and ROC area
                    fpr["micro"], tpr["micro"], _ = roc_curve(Y_test.ravel(), y_pred.ravel())
                    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

                    # First aggregate all false positive rates
                    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

                    # Then interpolate all ROC curves at this points
                    mean_tpr = np.zeros_like(all_fpr)
                    for i in range(n_classes):
                        mean_tpr += interp(all_fpr, fpr[i], tpr[i])

                    # Finally average it and compute AUC
                    mean_tpr /= n_classes

                    fpr["macro"] = all_fpr
                    tpr["macro"] = mean_tpr
                    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

                    # Plot all ROC curves
                    plt.figure()
                    lw = 2
                    plt.plot(fpr["micro"], tpr["micro"],
                             label='micro-average ROC curve (area = {0:0.2f})'
                                   ''.format(roc_auc["micro"]),
                             color='deeppink', linestyle=':', linewidth=4)

                    plt.plot(fpr["macro"], tpr["macro"],
                             label='macro-average ROC curve (area = {0:0.2f})'
                                   ''.format(roc_auc["macro"]),
                             color='navy', linestyle=':', linewidth=4)

                    colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
                    for i, color in zip(range(n_classes), colors):
                        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                                 label='ROC curve of class {0} (area = {1:0.2f})'
                                       ''.format(i, roc_auc[i]))

                    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
                    plt.xlim([0.0, 1.0])
                    plt.ylim([0.0, 1.05])
                    plt.xlabel('False Positive Rate')
                    plt.ylabel('True Positive Rate')
                    plt.title('Some extension of Receiver operating characteristic to multi-class')
                    plt.legend(loc="lower right")
                    plt.show()
                    pause()

                    ""

                    fold_auc_result.append(auc)
                    fold_result = sum(fold_auc_result) / kfold
                    parameter = [unit, opti, batch, drop]
                    grid_search.append(parameter)
                    print("fold_result", fold_result)

                print("grid_search",grid_search)
                print("fold_auc_result",fold_auc_result)
                pause()


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

        # Y_train = Y_train.values.ravel().astype(int)
        # Y_test = Y_test.values.ravel().astype(int)

        # Normalization
        X_standardScaler = preprocessing.StandardScaler()
        # 평균이 0, 표준편차가 1이 되도록 변환. #https://datascienceschool.net/view-notebook/f43be7d6515b48c0beb909826993c856/
        #                                .MinMaxScaler(range=(0, 1))
        X_tr_nor = X_standardScaler.fit_transform(X_train)
        X_te_nor = X_standardScaler.fit_transform(X_test)

        print(X_tr_nor.shape)

        model = Sequential()  # https://gist.github.com/NiharG15/cd8272c9639941cf8f481a7c4478d525
        model.add(Dense(units=unit, activation='relu', input_dim=X_tr_nor.shape[1]))  # input layer
        model.add(Dropout(drop))
        model.add(Dense(units=unit, activation='relu'))  # first hidden layer
        model.add(Dropout(drop))
        model.add(Dense(units=Y_train.shape[1], activation='softmax'))  # output layer
        # unit이 1이면,     activation = 'sigmoid' --> loss = 'binary_crossentropy'
        # softmax보다 sigmoid가 더 좋은 성능을 보임
        model.compile(optimizer=opti, loss='categorical_crossentropy', metrics=['accuracy', 'mse'])

        # 확인해주세요~~  에러만 안 나면 됩니다~~
        ##model.fit(X_tr_nor, Y_train, epochs=batch, batch_size=batch)

        model_save_path = '/home/kym/Otorhinolaryngology/result/Model/'  # GPU 경로
        model_name = '%s_StdZeroSoftL1_model.h5' % str(fold_num)
        model_save = os.path.join(model_save_path, model_name)
        checkpoint = [ModelCheckpoint(filepath=model_save, save_best_only=True, monitor='val_loss')]
        # val_mean_squared_error가 낮은 경우보다 val_loss를 저장했을 때 더 좋은 성능을 보임.

        history = model.fit(X_tr_nor, Y_train, epochs=epoch, batch_size=best_batch,
                            validation_data=(X_te_nor, Y_test),
                            callbacks=checkpoint, verbose=1)

        # weight 저장
        model_save_path = '/home/kym/Otorhinolaryngology/result/Model/'  # GPU 경로
        weight_name = '%s_StdZeroSoftL1_model_weight' % str(fold_num)
        weight_save = os.path.join(model_save_path, weight_name)
        model.save_weights(weight_save)

        saved_model = load_model(model_save)
        saved_weight = saved_model.load_weights(weight_save)

        y_pred = saved_model.predict(X_te_nor)

        # model validation
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

    end_time = now.strftime('%H:%M:%S')
    save_result1 = pd.DataFrame({'auc': best_auc_result,
                                 'start_time': start_time,
                                 'end_time': end_time})
    save_param1 = param_df

    log_dir = '/home/kym/Otorhinolaryngology/result/NeuralNet/10permutation/'  # GPU 경로
    os.chdir(log_dir)  # 경로 변경
    save_result1.to_csv('Std_zero_soft_layer1_auc_10per(191202).csv')
    save_param1.to_csv('Std_zero_soft_layer1_param_10per(191202).csv')
