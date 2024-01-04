import datetime
import logging
import math
import os
import random
import time
from collections import namedtuple
from pprint import pprint
from sklearn.metrics import classification_report
import joblib
import numpy as np
import pymysql
import torch
import keras.backend as K
from scipy.sparse import coo_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.utils import class_weight
from torch.nn import Parameter
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
from Data import read_file, Data, splitDataset

def trans_to_device(variable):  # Use the torch.cuda.is_available() function to check
    if torch.cuda.is_available():
        return variable.cuda()  # gpu
    else:
        return variable.cpu()  # cpu


def getVobab(domainnames: list[str]):
    """
    Glossary extracted from the list of domain names
    Save the corresponding characters to the dictionary
    :param domainnames:
    :return:
    """
    res = dict()
    for domainname in domainnames:
        for char in domainname:
            if char not in res:
                res[char] = len(res)
    return res



class HGATLayer(nn.Module):  # The alpha value is the balancing factor
    def __init__(self, in_features, out_features, dropout, alpha, transfer, concat=True, bias=False):
        super(HGATLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat
        self.transfer = transfer
        if self.transfer:
            self.weight = Parameter(torch.Tensor(self.in_features, self.out_features))
        else:
            self.register_parameter("weight", None)

        self.weight2 = Parameter(torch.Tensor(self.in_features, self.out_features))
        self.weight3 = Parameter(torch.Tensor(self.out_features, self.out_features))

        if bias:
            self.bias = Parameter(torch.Tensor(self.out_features))
        else:
            self.register_parameter("bias", None)

        self.word_context = nn.Embedding(1, self.out_features)  # Shape 1 x out_features

        self.a = nn.Parameter(torch.zeros(size=(2 * out_features, 1)))  # Xi model parameters
        self.a2 = nn.Parameter(torch.zeros(size=(2 * out_features, 1)))
        self.leakeyrelu = nn.LeakyReLU(self.alpha)

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.out_features)
        if self.weight is not None:
            self.weight.data.uniform_(-stdv, stdv)
        self.weight2.data.uniform_(-stdv, stdv)
        self.weight3.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

        nn.init.uniform_(self.a.data, -stdv, stdv)
        nn.init.uniform_(self.a2.data, -stdv, stdv)
        nn.init.uniform_(self.word_context.weight.data, -stdv, stdv)

    def forward(self, x, adj):
        '''
        :param x: Represents a node feature
        :param adj:Adjacency matrices
        :return:
        '''

        # print("\n", x.shape)
        x_4att = x.matmul(self.weight2)

        if self.transfer:  # For the subsequent mapping, the feature dimension is transformed first
            x = x.matmul(self.weight)
            if self.bias is not None:
                x = x + self.bias

        adj = adj.t()
        N1 = adj.shape[0]  # The number of edges
        N2 = adj.shape[1]  # Number of nodes

        # Calculates the attention weights of the edges
        pair = adj.nonzero().t()  # Gets a pair of non-zero elements
        x1 = x_4att[adj.nonzero().t()[1]]
        q1 = self.word_context.weight[0:].view(1, -1).repeat(x1.shape[0], 1)
        pair_h = torch.cat((q1, x1), dim=-1)
        pair_e = self.leakeyrelu(torch.matmul(pair_h, self.a).squeeze()).t()
        assert not torch.isnan(pair_e).any()
        pair_e = F.dropout(pair_e, self.dropout, training=self.training)
        e = torch.sparse_coo_tensor(pair, pair_e, torch.Size([N1, N2])).to_dense()
        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention_edge = F.softmax(attention, dim=-1)
        edge = torch.matmul(attention_edge, x)
        edge = F.dropout(edge, self.dropout, training=self.training)
        # Compute node attention
        edge_4att = edge.matmul(self.weight3)  # Maps the representation of the edge to a new representation space
        y1 = edge_4att[adj.nonzero().t()[0]]  # Gets the feature representation of the two nodes connected by each edge
        q1 = x_4att[adj.nonzero().t()[1]]
        pair_h = torch.cat((q1, y1), dim=-1)  # Stitches together the feature representations of the two connected nodes
        pair_e = self.leakeyrelu(torch.matmul(pair_h, self.a2).squeeze()).t()  # Calculates the attention weights of the edges
        assert not torch.isnan(pair_e).any()
        pair_e = F.dropout(pair_e, self.dropout, training=self.training)
        e = torch.sparse_coo_tensor(pair, pair_e, torch.Size([N1, N2])).to_dense()
        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention_node = F.softmax(attention.transpose(0, 1), dim=-1)  # The attention weight of each node to the other nodes is calculated.
        node = torch.matmul(attention_node, edge)
        if self.concat:
            node = F.elu(node)
        return node

    def __repr__(self):
        return self.__class__.__name__ + "（" + str(self.in_features) + " -> " + str(self.out_features) + "）"


class HGAT(nn.Module):
    def __init__(self, input_size, num_hidden, output_size, num_class, dropout=0.2):
        super(HGAT, self).__init__()
        self.dropout = dropout
        self.layer1 = HGATLayer(input_size, num_hidden, dropout=self.dropout, alpha=0.2, transfer=False, concat=True)
        self.layer2 = HGATLayer(num_hidden, output_size, dropout=self.dropout, alpha=0.2, transfer=True, concat=True)
        self.linear = nn.Linear(output_size, num_class)

    def forward(self, x, H):
        x = self.layer1(x, H)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.layer2(x, H)
        x = F.dropout(x , self.dropout, training=self.training)
        x = self.linear(x)
        return x

class AttentionLayer(nn.Module):
    def __init__(self, input_size):
        super(AttentionLayer, self).__init__()
        self.query = nn.Linear(input_size, input_size)
        self.key = nn.Linear(input_size, input_size)
        self.value = nn.Linear(input_size, input_size)

    def forward(self, x):
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        # Calculate attention weights
        attention_weights = F.softmax(torch.matmul(q, k.transpose(-2, -1)), dim=-1)

        # Weighted fusion
        attention_output = torch.matmul(attention_weights, v)

        return attention_output

class DomainGraph(nn.Module):
    def __init__(self, n_node, args: dict, class_weights, n_categories):
        super(DomainGraph, self).__init__()
        self.hidden_size = args.get("hiddenSize")
        self.n_node = n_node
        self.n_categories = n_categories
        self.batch_size = args.get("batchSize")
        self.dropout = args.get("dropout")
        self.initial_feature = args.get("initialFeatureSize")
        self.normalization = args.get("normalization")
        # self.dataset = args.get("dataset")

        self.embedding = nn.Embedding(self.n_node + 1, self.initial_feature, padding_idx=0)  # 将节点映射到初始特征大小的向量空间中
        self.layer_normH = nn.LayerNorm(self.hidden_size, eps=1e-6)
        self.attention_layer=AttentionLayer(self.hidden_size)
        if self.normalization:
            self.layer_normC = nn.LayerNorm(self.n_categories, eps=1e-6)
        self.prediction_transform = nn.Linear(self.hidden_size, self.n_categories, bias=True)
         # Extract the encoding layer of domain name character features
        # self.W1 = nn.Parameter(torch.Tensor(self.initial_feature, 1))
        self.conv1_3 = nn.Conv1d(self.initial_feature, self.initial_feature * 2, 3, padding=1) 
        self.conv1_5 = nn.Conv1d(self.initial_feature, self.initial_feature * 2, 5, padding=2)
        self.conv1_7 = nn.Conv1d(self.initial_feature, self.initial_feature * 2, 7, padding=3)
        self.relu = nn.LeakyReLU()
        self.ln1 = nn.Linear(Data.max_domainname_len * 3, Data.max_domainname_len)  
        self.conv2 = nn.Conv1d(self.initial_feature * 2, self.initial_feature * 4, 3, padding=1)
        self.conv3 = nn.Conv1d(self.initial_feature * 4, self.initial_feature * 8, 3, padding=1)
        self.conv4 = nn.Conv1d(self.initial_feature * 8, self.initial_feature * 16, 3, padding=1)
        self.conv5 = nn.Conv1d(self.initial_feature * 16, self.initial_feature * 32, 3, padding=1)
        self.ln2 = nn.Linear(self.initial_feature * 32, 1)
        self.fea_next = nn.Linear(self.n_categories, self.initial_feature)

        self.reset_parameters()
        self.network = HGAT(self.initial_feature, self.initial_feature, self.hidden_size, dropout=self.dropout,
                            num_class=self.n_categories)
        self.transformer_layer = nn.TransformerEncoderLayer(
            d_model=self.initial_feature,
            nhead=args.get("num_heads",1),
            dim_feedforward=self.hidden_size,
            dropout=self.dropout
        )
        self.transformer = nn.TransformerEncoder(self.transformer_layer, num_layers=args.get("num_layers"))
        self.class_weights = class_weights
        self.loss_function = nn.CrossEntropyLoss(weight=trans_to_device(torch.Tensor(self.class_weights).float()))
        self.optimizer = torch.optim.Adam(self.parameters(), lr=args.get("learning_rate"), weight_decay=args.get("l2"))
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=args.get("learning_rate_dc_step"),
                                                         gamma=args.get("learning_rate_dc"))

    def compute_scores(self, inputs):
        b = self.layer_normH(inputs)
        b = self.prediction_transform(b)
        pred = b

        if self.normalization:
            pred = self.layer_normC(b)
        return pred

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, X, G, last_feas=None):
        # print(X.shape)
        node=list()
        hidden = self.embedding(X)  
        hidden = hidden.permute(0, 2, 1)
        hidden1_3 = self.relu(self.conv1_3(hidden))
        hidden1_5 = self.relu(self.conv1_5(hidden))
        hidden1_7 = self.relu(self.conv1_7(hidden)) 
        hidden1 = torch.cat([hidden1_3, hidden1_5, hidden1_7], dim=-1)
        hidden1 = self.relu(self.ln1(hidden1))
        hidden2 = self.relu(self.conv2(hidden1))
        hidden3 = self.relu(self.conv3(hidden2))
        hidden4 = self.relu(self.conv4(hidden3))
        hidden5 = self.relu(self.conv5(hidden4))
        hidden5 = hidden5.permute(0, 2, 1)
        hidden = self.relu(self.ln2(hidden5)).squeeze()
        attention_output = self.attention_layer(hidden)
        hidden=hidden+attention_output
        if last_feas is not None and len(last_feas[0]) != len(hidden[0]):  
            hidden = hidden + last_feas.data
        nodes = self.network(hidden, G)
        node.append(nodes)
        if len(node) >= 2:
            transformer_input = torch.cat(node[-2:], dim=1)
            transformer_input = transformer_input.unsqueeze(0)
            transformer_output = self.transformer(transformer_input)
            nodes = transformer_output.squeeze(0)
        next_feas = self.relu(self.fea_next(nodes))

        return nodes, next_feas


params_global = {
    "batchSize": 256,
    "dropout": 0.1,
    "initialFeatureSize": 100,
    "normalization": True,
    "hiddenSize": 100,
    "num_layers": 3,  
    "learning_rate_dc_step": 700,
    "learning_rate": 1e-3,
    "learning_rate_dc": 0.5,
    "l2": 1e-3,
    "epoch": 2
}
 # Save some global parameters
if __name__ == '__main__':

    # Set the log output format
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(filename)s-%(funcName)s -  %(message)s",
        datefmt="%Y/%m/%d %H:%M:%S",
        level=logging.INFO
    )
    # Fixed random number seeds
    seed_value = 20230904
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)
    logging.info("The random number seed is set to：" + str(seed_value))
    # Device information
    if torch.cuda.is_available():
        assert torch.cuda.device_count() == 1  # Multiple GPUs won't work
        params_global["device"] = torch.device("cuda")
        logging.info("Choose to use a graphics card for training, GPU is" + torch.cuda.get_device_name(0))
    else:
        params_global["device"] = torch.device("cpu")
        logging.info("Select Use CPU for training")

    # 1、The data is read from the database and saved as a file
    # tablenames = ["a_train_malicious", "a_train_white", "a_test_malicious", "a_test_white", ]
    # for tablename in tablenames:
    #     print("form", tablename, "Start processing")
    #     data = getAllDataFromDBAndSave(tablename)
    #     saveDictToJson(data, os.path.join("./data/", tablename + ".json"))

    # 2、Read the data and build the features needed for a neural network
    # *******************************Subject***********************************
    data_rate = 1
    data_white = read_file("./data/a_train_white.json")
    data_mal = read_file("./data/a_train_malicious.json")
    data = data_white[:int(len(data_white) * data_rate)] + data_mal[:int(len(data_mal) * data_rate)]
    data2_rate = 0.1
    data_puretest_white = read_file("./data/a_test_white.json")
    data_puretest_mal = read_file("./data/a_test_malicious.json")
    data2 = data_puretest_white[:int(len(data_white) * data2_rate)] + data_puretest_mal[
                                                                      :int(len(data_mal) * data2_rate)]
    # print(data)
    # domainnames = [item.domainname for item in data+data2]
    domainnames = [item.domainname for item in data]



    vocab = getVobab(domainnames)
    print(vocab)
    # Training, validation, and testing are divided into 8:1:1
    # train_data, val_data, test_data = splitDataset(data, [0.8, 0.1, 0.1])
    # Divide the dataset according to 7:3
    train_data, test_data = splitDataset(data, [0.8, 0.2])
    # train_data, test_data = splitDataset(data+data2, [0.8, 0.2])
    # Calculate the labeling situation
    labels_idx = dict()
    labels_all = list()
    for item in data:
        if item.label not in labels_idx:
            labels_idx[item.label] = len(labels_idx)
        labels_all.append(item.label)

    # The weights are calculated, and they are used for later classification results
    class_weights = class_weight.compute_class_weight(class_weight='balanced', classes=np.unique(labels_all),
                                                      y=labels_all)
    # Construct the dataset
    train_data = Data(train_data, label_dict=labels_idx)
    # print(train_data.data)
    # val_data = Data(val_data, label_dict=labels_idx)
    test_data = Data(test_data, label_dict=labels_idx)
    # print(test_data)
    data2 = Data(data2, label_dict=labels_idx)

    best_loss = float('inf')
    best_model_weights = None
    # print(model)



    for epoch in range(3):
        model = trans_to_device(DomainGraph(args=params_global, class_weights=class_weights, n_categories=len(labels_idx),
            n_node=len(vocab)) )
        # pretrained_weights = torch.load('best_model_weights.pth')
        # model.load_state_dict(pretrained_weights)
        for epoch in range(params_global.get("epoch")):
            #
            print("*****************************************************")
            print("epoch:", epoch)

            print("Start training：", datetime.datetime.now())
            model.train()
            train_loss = 0.0
            slices = train_data.generate_batch(params_global.get("batchSize"))
            last_feas = None
            labels_all=[]
            preds_all=[]
            for step in tqdm(range(len(slices)), total=len(slices), ncols=70, leave=False, unit="b"):
                i = slices[step]
                X, X_mask, G, labels = train_data.get_slice(i, vocab)
                # X, X_mask, G, labels = train_data.get_slice_dt(i, vocab, clf)
                labels_number = trans_to_device(torch.LongTensor([labels_idx.get(label_id) for label_id in labels]))
                model.optimizer.zero_grad()
                X = trans_to_device(torch.Tensor(X).long())
                G = trans_to_device(torch.Tensor(G).float())
                preds, last_feas = model(X, G, last_feas)
                loss = model.loss_function(preds, labels_number)
                loss.backward()
                model.optimizer.step()
                model.scheduler.step()
                train_loss += loss
                labels_all.extend(labels_number.cpu().numpy())
                preds_all.extend(torch.argmax(preds, dim=-1).cpu().numpy())
                if loss < best_loss:
                    best_loss = loss
                    best_model_weights = model.state_dict().copy()
            acc = accuracy_score(labels_all, preds_all)
            pre = precision_score(labels_all, preds_all,
                                      zero_division=1)
            rec = recall_score(labels_all, preds_all,
                                   zero_division=1, average='weighted')
            f1 = f1_score(labels_all, preds_all,
                              average='weighted')


            print("acc:", acc)
            print("pre:", pre)
            print("rec:", rec)
            print("f1:", f1)
            # torch.save(model.state_dict(), f'model_epoch_{epoch}.pth')
            # print("--------------------------------------------------")
            print("epoch：", epoch, ",The total training dataset loss:", train_loss.item())
        # Here's the test
        torch.save(best_model_weights, 'best_model_weights.pth')
        test_split = 12
        slices_test = data2.generate_batch(params_global.get("batchSize"))
        slices_step_len = int(len(slices_test) / test_split)
        slices_steps = list()
        test_pred = list()
        test_labels = list()
        for i in range(test_split):
            slices_steps.append(slices_test[i * slices_step_len:i * slices_step_len + slices_step_len])
        for index, slices in enumerate(slices_steps):
                # Conduct model testing
            model.eval()
            last_feas = None
            time1 = time.time()
            for step in tqdm(range(len(slices)), total=len(slices), ncols=70, leave=False, unit="b"):
                try:
                    i = slices[step]
                    X, X_mask, G, labels = test_data.get_slice(i, vocab)
                    labels_number = trans_to_device(torch.LongTensor([labels_idx.get(label_id) for label_id in labels]))
                    X = trans_to_device(torch.Tensor(X).long())
                    G = trans_to_device(torch.Tensor(G).float())
                    preds, last_feas = model(X, G, last_feas)
                    test_pred.extend(torch.argmax(preds, dim=-1).cpu().numpy())
                    test_labels.extend(labels_number.cpu().numpy())
                except Exception as e:
                    print(f"Error in index {index}, step {step}: {e}")
                    raise e
            time2 = time.time()
            print("Test set time：", time2 - time1)
            # print(test_pred,test_labels)
        acc = accuracy_score(test_labels, test_pred)
        pre = precision_score(test_labels, test_pred, average='weighted')
        rec = recall_score(test_labels, test_pred, average='weighted')
        f1 = f1_score(test_labels, test_pred, average='weighted')
        print("acc:", acc)
        print("pre:", pre)
        print("rec:", rec)
        print("f1:", f1)
        print("--------------------------------------------------")
        report=classification_report(test_labels,test_pred, output_dict=True)
        # print(class_weights)
        # print(report)
        f1_scores = [report[str(i)]['f1-score'] for i in range(2)]
        x=0
        for scores in f1_scores:
            f1_scores[x] = 1 / (scores + K.epsilon())
            x += 1
        # print(f1_scores)
        for a in range(2):
            class_weights[a] = class_weights[a] + f1_scores[a]
        
