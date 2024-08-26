import os

foldi = 1
for i in range(5):
    datapath = './mydataset/5fold/tsRNA-disease-fold%d/' % foldi
    featurepath = './Feature/5fold/fold%d_embedding_feature.h5' % foldi

    print('foldi', foldi)
    foldi += 1
    cmd = "python main.py --data_path %s  --FeaturePath %s" % (datapath, featurepath)

    os.system(cmd)
print("5fold finish")