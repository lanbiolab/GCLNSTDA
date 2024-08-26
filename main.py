import numpy as np
import torch
from model import GCLNSTDA

from utils import metrics, scipy_sparse_mat_to_torch_sparse_tensor
from parser1 import args
from tqdm import tqdm
import torch.utils.data as data
from utils import TrnData

from preprocessing import dataload

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

d = args.d
l = args.gnn_layer
temp = args.temp
batch_user = args.batch
epoch_no = args.epoch
max_samp = 40
lambda_1 = args.lambda1
lambda_2 = args.lambda2
dropout = args.dropout
lr = args.lr
decay = args.decay
svd_q = args.q
feature_path = args.FeaturePath
feature_path_backups = args.FeaturePath_backups


train, test, prior, uninter_mat = dataload(args.data_path)
train_csr = (train != 0).astype(np.float32)
print('Data loaded.')

print('user_num:', train.shape[0], 'item_num:', train.shape[1], 'lambda_1:', lambda_1, 'lambda_2:', lambda_2, 'temp:',
      temp, 'q:', svd_q)

epoch_user = min(train.shape[0], 30000)

# normalizing the adj matrix
rowD = np.array(train.sum(1)).squeeze()
colD = np.array(train.sum(0)).squeeze()
for i in range(len(train.data)):
    train.data[i] = train.data[i] / pow(rowD[train.row[i]]*colD[train.col[i]], 0.5)

# construct data loader
train = train.tocoo()
train_data = TrnData(train)
train_loader = data.DataLoader(train_data, batch_size=args.inter_batch, shuffle=True, num_workers=0)

adj_norm = scipy_sparse_mat_to_torch_sparse_tensor(train)
adj_norm = adj_norm.coalesce()
print('Adj matrix normalized.')
print('adj_norm.shape', adj_norm.shape)

# perform svd reconstruction
adj = scipy_sparse_mat_to_torch_sparse_tensor(train).coalesce()
print('Performing SVD...')
svd_u, s, svd_v = torch.svd_lowrank(adj, q=svd_q)
u_mul_s = svd_u @ (torch.diag(s))
v_mul_s = svd_v @ (torch.diag(s))
del s
print('SVD done.')

test_labels = [[] for i in range(test.shape[0])]
for i in range(len(test.data)):
    row = test.row[i]
    col = test.col[i]
    test_labels[row].append(col)
print('Test data processed.')

loss_list = []
loss_r_list = []
loss_s_list = []
recall_20_x = []
recall_20_y = []
ndcg_20_y = []
recall_40_y = []
ndcg_40_y = []

model = GCLNSTDA(adj_norm.shape[0], adj_norm.shape[1], d, u_mul_s, v_mul_s, svd_u.T, svd_v.T, train_csr, adj_norm, l,
                 temp, lambda_1, lambda_2, dropout, batch_user, feature_path, feature_path_backups, prior, uninter_mat,
                 device)
optimizer = torch.optim.Adam(model.parameters(), weight_decay=0, lr=lr)

current_lr = lr

for epoch in range(epoch_no):

    epoch_loss = 0
    epoch_loss_r = 0
    epoch_loss_s = 0
    train_loader.dataset.neg_sampling()
    for i, batch in enumerate(tqdm(train_loader)):
        uids, pos, neg = batch
        uids = uids.long()
        pos = pos.long()
        neg = neg.long()
        iids = torch.concat([pos, neg], dim=0)

        optimizer.zero_grad()
        loss, loss_r, loss_s = model(uids, iids, pos, neg)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.cpu().item()
        epoch_loss_r += loss_r.cpu().item()
        epoch_loss_s += loss_s.cpu().item()

        torch.cuda.empty_cache()

    batch_no = len(train_loader)
    epoch_loss = epoch_loss/batch_no
    epoch_loss_r = epoch_loss_r/batch_no
    epoch_loss_s = epoch_loss_s/batch_no
    loss_list.append(epoch_loss)
    loss_r_list.append(epoch_loss_r)
    loss_s_list.append(epoch_loss_s)
    print('Epoch:', epoch, 'Loss:', epoch_loss, 'Loss_r:', epoch_loss_r, 'Loss_s:', epoch_loss_s)

test_uids = np.array([i for i in range(adj_norm.shape[0])])

test_uids_input = torch.LongTensor(test_uids)
model(test_uids_input, None, None, None, test=True)
