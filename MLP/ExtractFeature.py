import h5py
import numpy as np
import matplotlib.pyplot as plt
# from tensorflow.python.keras.models import Sequential
# from tensorflow.python.keras.layers import Dense, Dropout
# from tensorflow.python.keras.optimizers import Adam
# from tensorflow.python.keras.regularizers import l2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from keras.regularizers import l2

import sortscore
import time

num_tsRNA = 296
num_disease = 74

all_tpr = []
all_fpr = []
all_recall = []
all_precision = []
all_accuracy = []
all_F1 = []
foldi=1

root_path = '../'
for ii in range(5):
    FeaturePath = root_path + 'Feature/5fold/fold%d_embedding_feature.h5' % foldi
    trainfile_path = root_path + 'mydataset/5fold/tsRNA-disease-fold%d/train.txt' % foldi
    testfile_path = root_path + 'mydataset/5fold/tsRNA-disease-fold%d/test.txt' % foldi

    foldi += 1

    train_exist_users = []
    train_exist_items = []
    with open(trainfile_path) as f:
        for l in f.readlines():
            if len(l) > 0:
                l = l.strip('\n').split(' ')
                items = [int(i) for i in l[1:]]
                uid = int(l[0])
                train_exist_users.append(uid)
                train_exist_items.append(items)

    train_u_nodes = []
    train_v_nodes = []
    train_index = []

    for i in range(len(train_exist_users)):
        temp_user = train_exist_users[i]
        for j in range(len(train_exist_items[i])):
            temp_item = train_exist_items[i][j]
            train_u_nodes.append(temp_user)
            train_v_nodes.append(temp_item)
            train_index.append((temp_user, temp_item))

    test_exist_users = []
    test_exist_items = []
    with open(testfile_path) as f:
        for l in f.readlines():
            if len(l) > 0:
                l = l.strip('\n').split(' ')
                items = [int(i) for i in l[1:]]
                uid = int(l[0])
                test_exist_users.append(uid)
                test_exist_items.append(items)

    test_u_nodes = []
    test_v_nodes = []
    test_index = []

    for i in range(len(test_exist_users)):
        temp_user = test_exist_users[i]
        for j in range(len(test_exist_items[i])):
            temp_item = test_exist_items[i][j]
            test_u_nodes.append(temp_user)
            test_v_nodes.append(temp_item)
            test_index.append((temp_user, temp_item))

    tsrna_disease_matrix = np.zeros((num_tsRNA, num_disease))
    for i in range(len(train_exist_users)):
        temp_user = train_exist_users[i]
        for j in range(len(train_exist_items[i])):
            temp_item = train_exist_items[i][j]
            tsrna_disease_matrix[temp_user][temp_item] = 1

    for i in range(len(test_exist_users)):
        temp_user = test_exist_users[i]
        for j in range(len(test_exist_items[i])):
            temp_item = test_exist_items[i][j]
            tsrna_disease_matrix[temp_user][temp_item] = 1

    with h5py.File(FeaturePath, 'r') as f:
        gcn_tsRNA_feature = f['user_feature'][:]
        gcn_disease_feature = f['item_feature'][:]


    new_tsrna_disease_matrix = tsrna_disease_matrix.copy()
    for index in test_index:
        new_tsrna_disease_matrix[index[0], index[1]] = 0
    roc_tsrna_disease_matrix = new_tsrna_disease_matrix+tsrna_disease_matrix
    rel_matrix = new_tsrna_disease_matrix

    input_fusion_feature_x = []
    input_fusion_x_label=[]
    for (u,i) in train_index:
        gcn_tsRNA_array = gcn_tsRNA_feature[u,:]
        gcn_disease_array = gcn_disease_feature[i,:]
        fusion_feature = np.concatenate((gcn_tsRNA_array, gcn_disease_array), axis=0)
        input_fusion_feature_x.append(fusion_feature.tolist())
        input_fusion_x_label.append(1)
        for num in range(1):
            j = np.random.randint(rel_matrix.shape[1])
            while (u,j) in train_index:
                j = np.random.randint(rel_matrix.shape[1])
            gcn_disease_array = gcn_disease_feature[j,:]
            fusion_feature = np.concatenate((gcn_tsRNA_array, gcn_disease_array), axis=0)
            input_fusion_feature_x.append(fusion_feature.tolist())
            input_fusion_x_label.append(0)

    input_fusion_feature_test_x=[]
    input_fusion_test_x_label=[]
    for row in range(rel_matrix.shape[0]):
        for col in range(rel_matrix.shape[1]):
            gcn_tsRNA_array = gcn_tsRNA_feature[row, :]
            gcn_disease_array = gcn_disease_feature[col,:]

            fusion_feature = np.concatenate((gcn_tsRNA_array, gcn_disease_array), axis=0)

            input_fusion_feature_test_x.append(fusion_feature.tolist())
            input_fusion_test_x_label.append(rel_matrix[row,col])

    model = Sequential()

    model.add(Dense(256, input_shape=(256,), kernel_regularizer=l2(0.0001), activation='relu', name='dense1'))
    model.add(Dropout(0.3))
    model.add(Dense(128, activation='relu', kernel_regularizer=l2(0.0001), name='dense2'))
    model.add(Dropout(0.3))
    model.add(Dense(32, activation='relu', kernel_regularizer=l2(0.0001), name='dense3'))
    model.add(Dropout(0.3))
    model.add(Dense(1, activation='sigmoid', kernel_initializer='glorot_normal', name='prediction'))
    model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
    history = model.fit(np.array(input_fusion_feature_x), np.array(input_fusion_x_label), epochs=300, batch_size=100)
    predictions = model.predict(np.array(input_fusion_feature_test_x), batch_size=10)
    prediction_matrix = np.zeros((rel_matrix.shape[0], rel_matrix.shape[1]))
    predictions_index = 0
    for row in range(prediction_matrix.shape[0]):
        for col in range(prediction_matrix.shape[1]):
            prediction_matrix[row, col] = predictions[predictions_index]
            predictions_index += 1
    aa = prediction_matrix.shape
    bb = roc_tsrna_disease_matrix.shape
    zero_matrix = np.zeros((prediction_matrix.shape[0], prediction_matrix.shape[1]))

    score_matrix_temp = prediction_matrix.copy()
    score_matrix = score_matrix_temp + zero_matrix
    minvalue = np.min(score_matrix)
    score_matrix[np.where(roc_tsrna_disease_matrix == 2)] = minvalue - 20
    sorted_tsrna_disease_matrix, sorted_score_matrix = sortscore.sort_matrix(score_matrix, roc_tsrna_disease_matrix)

    tpr_list = []
    fpr_list = []
    recall_list = []
    precision_list = []
    accuracy_list = []
    F1_list = []
    for cutoff in range(sorted_tsrna_disease_matrix.shape[0]):
        P_matrix = sorted_tsrna_disease_matrix[0:cutoff + 1, :]
        N_matrix = sorted_tsrna_disease_matrix[cutoff + 1:sorted_tsrna_disease_matrix.shape[0] + 1, :]
        TP = np.sum(P_matrix == 1)
        FP = np.sum(P_matrix == 0)
        TN = np.sum(N_matrix == 0)
        FN = np.sum(N_matrix == 1)
        tpr = TP / (TP + FN)
        fpr = FP / (FP + TN)
        tpr_list.append(tpr)
        fpr_list.append(fpr)
        recall = TP / (TP + FN)
        precision = TP / (TP + FP)
        recall_list.append(recall)
        precision_list.append(precision)
        accuracy = (TN + TP) / (TN + TP + FN + FP)
        F1 = (2 * TP) / (2 * TP + FP + FN)
        F1_list.append(F1)

        accuracy_list.append(accuracy)

    tpr_arr_epoch = np.array(tpr_list)
    fpr_arr_epoch = np.array(fpr_list)
    recall_arr_epoch = np.array(recall_list)
    precision_arr_epoch = np.array(precision_list)
    accuracy_arr_epoch = np.array(accuracy_list)
    F1_arr_epoch = np.array(F1_list)
    # print("epoch,",epoch)
    print("accuracy:%.4f,recall:%.4f,precision:%.4f,F1:%.4f" % (
        np.mean(accuracy_arr_epoch), np.mean(recall_arr_epoch), np.mean(precision_arr_epoch),
        np.mean(F1_arr_epoch)))
    print("roc_auc", np.trapz(tpr_arr_epoch, fpr_arr_epoch))
    print("AUPR", np.trapz(precision_arr_epoch, recall_arr_epoch))


    all_tpr.append(tpr_list)
    all_fpr.append(fpr_list)
    all_recall.append(recall_list)
    all_precision.append(precision_list)
    all_accuracy.append(accuracy_list)
    all_F1.append(F1_list)

tpr_arr = np.array(all_tpr)
fpr_arr = np.array(all_fpr)
recall_arr = np.array(all_recall)
precision_arr = np.array(all_precision)
accuracy_arr = np.array(all_accuracy)
F1_arr = np.array(all_F1)

mean_cross_tpr = np.mean(tpr_arr, axis=0)  # axis=0
mean_cross_fpr = np.mean(fpr_arr, axis=0)
mean_cross_recall = np.mean(recall_arr, axis=0)
mean_cross_precision = np.mean(precision_arr, axis=0)
mean_cross_accuracy = np.mean(accuracy_arr, axis=0)

mean_accuracy = np.mean(np.mean(accuracy_arr, axis=1), axis=0)
mean_recall = np.mean(np.mean(recall_arr, axis=1), axis=0)
mean_precision = np.mean(np.mean(precision_arr, axis=1), axis=0)
mean_F1 = np.mean(np.mean(F1_arr, axis=1), axis=0)

print("accuracy:%.4f,recall:%.4f,precision:%.4f,F1:%.4f"%(mean_accuracy, mean_recall, mean_precision, mean_F1))

roc_auc = np.trapz(mean_cross_tpr, mean_cross_fpr)
AUPR = np.trapz(mean_cross_precision, mean_cross_recall)

print("AUC:%.4f,AUPR:%.4f"%(roc_auc, AUPR))

folders = "./PlotFigure"


with h5py.File(folders + '/5fold_AUC.h5','w') as hf:
    hf['fpr'] = mean_cross_fpr
    hf['tpr'] = mean_cross_tpr
with h5py.File(folders +'/5fold_AUPR.h5','w') as h:
    h['recall'] = mean_cross_recall
    h['precision'] = mean_cross_precision


plt.plot(mean_cross_fpr, mean_cross_tpr, label='mean ROC=%0.4f' % roc_auc)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc=0)
plt.savefig(folders + "/5fold.png")
print("runtime over, now is :")
print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
plt.show()