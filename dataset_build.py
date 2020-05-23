import torch
import numpy as np
import torch.utils.data as Data
from keras.preprocessing.sequence import pad_sequences

pad_value = {'movie':0,'music':19981}

def load_data(args):
    ent_embed=np.load('../data/'+args.dataset+'/pretrained_embeddings/ent_embed'+str(200)+'.npy')
    item_attr_set = np.load('../data/' + args.dataset + '/pretrained_embeddings/item_attr_set.npy')
    train = np.load('../data/' + args.dataset + '/data/train.npy', allow_pickle=True).tolist()
    print(len(train))
    test=np.load('../data/'+args.dataset+'/data/test.npy', allow_pickle=True).tolist()[0:520650]
    args.num_attrs = torch.from_numpy(np.load('../data/'+args.dataset+'/pretrained_embeddings/attr_embed'+str(200)+'.npy')).cuda().size(0)

    if(args.dataset == 'music'):
        ent_embed=np.row_stack((ent_embed,ent_embed[-1]))
        item_attr_set = np.row_stack((item_attr_set, [13888] * 4))

    train = pad_sequences(train, maxlen=args.L + 1, dtype='int32', value=pad_value[args.dataset])
    args.num_items = torch.from_numpy(ent_embed).cuda().size(0)
    item_attr_set = torch.from_numpy(item_attr_set).float().cuda()
    train_seq = torch.from_numpy(train).long().cuda()
    train_set_length=len(train)
    train_label = torch.from_numpy(np.array([1,0,0,0,0] * int(len(train) / 5))).long().cuda()
    test_seq = torch.from_numpy(pad_sequences(test, maxlen=args.L+1, dtype='int32')).long().cuda()

    dataset = Data.TensorDataset(train_seq, train_label)

    loader = Data.DataLoader(dataset=dataset, batch_size=args.batch_size, shuffle=True)

    # args.num_items = ent_embed.size(0)
    # args.num_attrs = attr_embed.size(0)

    return loader, train_set_length, test_seq, item_attr_set
