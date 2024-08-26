
import numpy as np
import scipy.sparse as sp
from scipy.sparse import coo_matrix


def get_uninteracted_item(train_dict, num_users, num_items):
    all_items = list(range(num_items))
    uninteracted_dict = {}
    num_uninter = []
    for user in train_dict.keys():
        interacted_items = set(train_dict[user].keys())
        uninteracted_items = set(all_items) - interacted_items
        uninteracted_dict[user] = list(uninteracted_items)
        num_uninter.append(len(uninteracted_items))
    return uninteracted_dict, num_uninter

def get_prior(trainall_index):
    data_dict = {}
    datapair = []
    popularity = np.zeros(296)

    for i in trainall_index:
        user, item, rating = i[0],i[1], 1
        user, item = int(user), int(item)
        popularity[int(item)] += 1
        data_dict.setdefault(user, {})
        data_dict[user][item] = 1
        datapair.append((user, item))
    prior = popularity / sum(popularity)

    return data_dict, prior

def dataload(data_path):

    #tsRNA-disease
    num_users = 296
    num_items = 74
    train_exist_users = []
    train_exist_items = []
    trainfile_path = data_path + 'train.txt'
    testfile_path = data_path + 'test.txt'



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
    train_ratings = []
    train_index = []

    trainall_u_nodes = []
    trainall_v_nodes = []
    trainall_ratings = []
    trainall_index = []



    for i in range(len(train_exist_users)):
        temp_user = train_exist_users[i]
        for j in range(len(train_exist_items[i])):
            temp_item = train_exist_items[i][j]
            train_u_nodes.append(temp_user)
            train_v_nodes.append(temp_item)
            train_ratings.append(1)
            train_index.append((temp_user,temp_item))

            trainall_u_nodes.append(temp_user)
            trainall_v_nodes.append(temp_item)
            trainall_ratings.append(1)
            trainall_index.append((temp_user, temp_item))

    data_dict, prior = get_prior(trainall_index)
    uninter_mat, num_uninter = get_uninteracted_item(data_dict, num_users, num_items)



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
    test_ratings = []
    test_index = []

    for i in range(len(test_exist_users)):
        temp_user = test_exist_users[i]
        for j in range(len(test_exist_items[i])):
            temp_item = test_exist_items[i][j]
            test_u_nodes.append(temp_user)
            test_v_nodes.append(temp_item)
            test_ratings.append(1)
            test_index.append((temp_user, temp_item))


    nozero_test_index = test_index.copy()
    for i in range(num_users):
        for j in range(num_items):
            if (i, j) not in train_index and (i, j) not in nozero_test_index:
                test_u_nodes.append(i)
                test_v_nodes.append(j)
                test_ratings.append(0)
                test_index.append((i, j))


    train_ratings = np.array(train_ratings)
    test_ratings = np.array(test_ratings)



    train_pairs_nonzero = np.vstack([train_u_nodes, train_v_nodes]).transpose()
    test_pairs_nonzero = np.vstack([test_u_nodes, test_v_nodes]).transpose()

    train_pairs_idx = train_pairs_nonzero.copy()
    test_pairs_idx = test_pairs_nonzero.copy()

    u_test_idx, v_test_idx = test_pairs_idx.transpose()
    u_train_idx, v_train_idx = train_pairs_idx.transpose()


    train_all_labels = np.array(train_ratings, dtype=np.int32)
    test_all_labels = np.array(test_ratings, dtype=np.int32)



    train_labels = train_all_labels.copy()
    test_labels = test_all_labels.copy()

    u_train_idx = np.hstack([u_train_idx])
    v_train_idx = np.hstack([v_train_idx])
    train_labels = np.hstack([train_labels])

    train_data = train_labels
    train_data = train_data.astype(np.float32)

    test_data = test_labels
    test_data = test_data.astype(np.float32)


    train_coo = coo_matrix((train_data, (u_train_idx, v_train_idx)),shape=(num_users, num_items))
    test_coo = coo_matrix((test_data, (u_test_idx, v_test_idx)),shape=(num_users, num_items))

    return train_coo, test_coo, prior, uninter_mat