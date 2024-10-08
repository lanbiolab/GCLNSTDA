import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Model Params')
    parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
    parser.add_argument('--decay', default=0.99, type=float, help='learning rate')
    parser.add_argument('--batch', default=64, type=int, help='batch size')
    parser.add_argument('--inter_batch', default=32, type=int, help='batch size')
    parser.add_argument('--note', default=None, type=str, help='note')
    parser.add_argument('--lambda1', default=0.1, type=float, help='weight of cl loss')
    parser.add_argument('--epoch', default=100, type=int, help='number of epochs')
    parser.add_argument('--d', default=128, type=int, help='embedding size')
    parser.add_argument('--q', default=1, type=int, help='rank')
    parser.add_argument('--gnn_layer', default=2, type=int, help='number of gnn layers')
    parser.add_argument('--data', default='tsRNA_disease', type=str, help='name of dataset')
    parser.add_argument('--dropout', default=0.0, type=float, help='rate for edge dropout')
    parser.add_argument('--temp', default=0.2, type=float, help='temperature in cl loss')
    parser.add_argument('--lambda2', default=1e-7, type=float, help='l2 reg weight')
    parser.add_argument('--cuda', default='0', type=str, help='the gpu to use')
    parser.add_argument('--data_path', nargs='?', default='./mydataset/5fold/tsRNA-disease-fold1/',
                        help='Input data path.')
    parser.add_argument('--FeaturePath', nargs='?', default='./Feature/5fold/fold1_embedding_feature.h5',
                        help='Feature save path')
    parser.add_argument('--FeaturePath_backups', nargs='?', default='./Feature/5fold/fold1_embedding_feature.h5',
                        help='FeaturePath_backups save path')

    return parser.parse_args()

args = parse_args()
