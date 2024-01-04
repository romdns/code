import datetime
import logging
import math
from urllib import parse
import os
from math import log
import random
import time
import re
from collections import namedtuple
from pprint import pprint
import joblib
import numpy as np
import pymysql
import torch
from scipy.sparse import coo_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.utils import class_weight
from torch.nn import Parameter
from tqdm import tqdm

from featureUtils import saveDictToJson, readDictFromJson, featureExtract, calDomainNameFea
import torch.nn as nn
import torch.nn.functional as F
def extract_tld(domain):
    match = re.search(r'\.([a-zA-Z]+)$', domain)
    if match:
        return match.group(1)
    else:
        return None
def InfoEntropy(str_1):
    Info_map = {}

    for i in str_1:
        if i != ' ' and i != '"' and i != "." and i != ',':
            if i in Info_map.keys():
                Info_map[i] += 1
            else:
                Info_map[i] = 1

    return calcShannonEnt(Info_map)

def calcShannonEnt(dataSet):
    numEntries = 0
    shannonEnt = 0.0

    for key in dataSet:
        numEntries += dataSet[key]


    for key in dataSet:
        prob = float(dataSet[key]) / numEntries 
        shannonEnt -= prob * log(prob, 2) 
    return shannonEnt
def transDbFieldStrToList(field: str):
    """
    Converts the string read from the database to a list
    :param field:
    :param spliter:
    :return:
    """
    if not field:
        return None
    spliter = field[-1]
    if spliter == ";":
        return field[:-1].split(";")
    else:
        return field.split(",")


def read_file(filepath: str):
    """
    The main role is to construct structured data from files
    :param filepath:
    :return:
    """
    domainDataItemStruct = namedtuple("domainDataItemStruct",
                                      ["Country", "ASN", "TTL", "IP", "Domain", "State", "Registrant_Name", "Creation_Date_Time",
                                       "Domain_Name", "Alexa_Rank", "subdomain", "Organization", "len",
                                       "longest_word", "oc_32", "shortened", "obfuscate_at_sign",
                                       "entropy", "Domain_Age", "tld", "Emails",
                                       "numeric_percentage", "puny_coded", "typos", "oc_8",
                                       "char_distribution","Registrar", "sld","Name_Server_Count","Page_Rank","type"])
    res = list()
    context = readDictFromJson(filepath) # This is the complete original file in the library

    for id,line in enumerate(context):
        Domain = line[4]
        Country = line[0]
        ASN=line[1]
        TTL= line[2]
        IP=line[3]
        State = line[5]
        Registrant_Name = line[6]
        Creation_Date_Time = line[7]
        Domain_Name = line[8]
        Alexa_Rank = line[9]
        subdomain = line[10]
        Organization = line[11]
        len = line[12]
        longest_word = line[13]
        oc_32 = line[14]
        shortened  = line[15]
        obfuscate_at_sign  = line[16]
        entropy = line[17]
        Domain_Age = line[18]
        tld = line[19]
        Emails = line[20]
        numeric_percentage= line[21]
        puny_coded = line[22]
        typos = line[23]
        oc_8 = line[24]
        char_distribution = line[25]
        Registrar =line[26]
        sld=line[27]
        Name_Server_Count=line[28]
        Page_Rank=line[29]
        type=line[30]
        res.append(domainDataItemStruct(
            Domain=Domain,
            Country=Country,
            ASN=ASN,
            TTL=TTL,
            IP=IP,
            State=State,
            Registrant_Name=Registrant_Name,
            Creation_Date_Time=Creation_Date_Time,
            Domain_Name=Domain_Name,
            Alexa_Rank=Alexa_Rank,
            subdomain=subdomain,
            Organization=Organization,
            len=len,
            longest_word=longest_word,
            oc_32=oc_32,
             shortened = shortened ,
            obfuscate_at_sign =obfuscate_at_sign ,
            entropy=entropy,
            Domain_Age=Domain_Age,
            tld=tld,
            Emails=Emails,
            numeric_percentage=numeric_percentage,
            puny_coded=puny_coded,
            typos=typos,
            oc_8=oc_8,
            char_distribution=char_distribution,
            Registrar =Registrar,
            sld=sld,
            Name_Server_Count=Name_Server_Count ,
            Page_Rank=Page_Rank,
            type=type
        ))
    return res


def splitDataset(dataset, rate: list):
    """
     Divide the data set
    :param dataset:
    :param rate:
    :return:
    """
    if len(rate) <= 1:
        logging.error("Please set at least two divisions")
        return
    n_samples = int(len(dataset))
    sidx = np.arange(n_samples, dtype="int32")
    np.random.shuffle(sidx)
    res = list()
    last_p = 0
    for index in range(len(rate) - 1):
        cur_rate = rate[index]
        cur_number = int(cur_rate * n_samples)
        res.append([dataset[item] for item in sidx[last_p:last_p + cur_number]])
        last_p += cur_number
    res.append([dataset[item] for item in sidx[last_p:]])
    return res


class Data():
    """
    Customize the way datasets are generated
    """
    max_domainname_len = 100

    def __init__(self, data, label_dict):
        self.data = data
        self.label_map = label_dict
        self.data_len = len(data)

    def generate_batch(self, batch_size):
        n_batch = int(self.data_len / batch_size)
        if self.data_len % batch_size != 0:  # At this time, the sample cannot be completely divided into batches
            n_batch += 1
        slices = np.split(np.arange(n_batch * batch_size), n_batch)  # Divide the corresponding subscripts first
        slices[-1] = slices[-1][:(self.data_len - batch_size * (n_batch - 1))]  # Subtract the excess
        return slices
    def get_slice(self, iList, vocab: dict):
        try:
            inputdata = [self.data[item] for item in iList]
        except Exception as e:
            print("发生异常")
            print(f"iList: {iList}")
            print(f"Exception: {e}")
            return [], [], [], []  # Returns an empty list, which can be adjusted based on actual conditions

        X = list()  # Node features
        X_mask = list()
        H = np.array([])  # fig
        labels = list()  # Category labels

        attr_list = list()
        for attr in inputdata:
            domainname_new = list()
            for c in attr.Domain:
                domainname_new.append(vocab[c])
            Domain = domainname_new
             # Make up the maximum length of the domain name (0), or cut it out
            domain_len = len(Domain)
            if domain_len == 0:
                print(attr)
            line_mask = [1] * domain_len
            if domain_len < Data.max_domainname_len:
                padding_id = len(vocab)
                Domain.extend([padding_id] * (Data.max_domainname_len - domain_len))
                line_mask.extend([0] * (Data.max_domainname_len - domain_len))
            elif domain_len > Data.max_domainname_len:
                Domain = Domain[:Data.max_domainname_len]
            X.append(Domain)
            X_mask.append(line_mask)
            #  Calculate the labels
            labels.append(attr.type)

           # Attributes are appended directly to the back
            attrs = [attr.Country, attr.ASN, attr.TTL, attr.IP, attr.State, attr.Registrant_Name, attr.Creation_Date_Time,
                     attr.Domain_Name,attr.Alexa_Rank, attr.subdomain, attr.Organization, attr.len, attr.longest_word, attr.oc_32,
                     attr.shortened , attr.obfuscate_at_sign , attr.entropy,
                     attr.Domain_Age, attr.tld, attr.Emails, attr.numeric_percentage,
                     attr.puny_coded, attr.typos, attr.oc_8, attr.char_distribution, attr.Registrar, attr.sld, attr.Name_Server_Count,attr.Page_Rank]

            if not attr_list:
                attr_list = [[attr] for attr in attrs]
            else:
                for _index, _a in enumerate(attrs):
                    attr_list[_index].append(_a)

        # Composition from attributes
        for attr in attr_list:
            #This method returns information about the hyperedge of the property construction
            attrs = self.createHyperEdgeByAttr(attr)
            if not attrs:
                continue
            if H.size:
                H = np.hstack((H, attrs))
            else:
                H = np.array(attrs)

        # Add an edge of yourself and your node
        self_loop = np.eye(len(iList), len(iList))#Create an identity matrix
        H = np.hstack((H, self_loop))

        return X, X_mask, H, labels

    def get_slice_dt(self, iList, vocab: dict, clf):
        """
        Generate the data that needs to be portional
        :param iList:
        :param vocab: glossary
        :return:
        """
        try:
            inputdata = [self.data[item] for item in iList]
        except Exception as e:
            print("An exception has occurred",e)
            print(iList)
            return
        data_dt = list()
        for item in inputdata:
            line_tmp = list()
            line_tmp.extend(calDomainNameFea(item.domainname))
            line_tmp.append(len(item.ip4) if item.ip4 else 0)
            line_tmp.append(len(item.ip6) if item.ip6 else 0)
            line_tmp.append(len(item.cname) if item.cname else 0)
            line_tmp.append(len(item.ns) if item.ns else 0)
            line_tmp.append(len(item.mx) if item.mx else 0)
            line_tmp.append(len(item.soa_prins) if item.soa_prins else 0)
            line_tmp.append(len(item.soa_respmail_addr) if item.soa_respmail_addr else 0)
            line_tmp.append(item.soa_serial if item.soa_serial else 0)
            line_tmp.append(item.subdomain if item.subdomain else 0)
            line_tmp.append(item.Organization if item.Organization else 0)
            line_tmp.append(item.len if item.len else 0)
            line_tmp.append(item.longest_word if item.longest_word else 0)
            line_tmp.append(1 if item. shortened  else 0)
            line_tmp.append(1 if item.obfuscate_at_sign  else 0)
            line_tmp.append(2023 - int(item.entropy.split("-")[0]) if item.entropy else 0)
            line_tmp.append(int(item.whois_expire_date.split("-")[0]) if item.whois_expire_date else 0)
            line_tmp.append(1 if item.whois_last_update else 0)
            line_tmp.append(len(item.Emails) if item.Emails else 0)
            line_tmp.append(len(item.Emailses) if item.Emailses else 0)
            line_tmp.append(len(item.puny_coded) if item.puny_coded else 0)
            line_tmp.append(len(item.whois_name_servers) if item.whois_name_servers else 0)
            line_tmp.append(len(item.oc_8) if item.oc_8 else 0)
            # line_tmp.append(len(item.whois_email) if item.whois_email else 0)
            data_dt.append(line_tmp)

        paths = clf.decision_path(data_dt)
        # paths, _ = clf.decision_path(data_dt)
        edgeslist = divideToGroupByDT(paths)
        H = self.createEdgeFromFeatureDT(len(iList), edgeslist)

        X = list()  # Node features
        X_mask = list()
        labels = list()  # Category labels

        for attr in inputdata:
            domainname_new = list()
            for c in attr.domainname:
                if c in vocab:
                    domainname_new.append(vocab[c])
                else:
                    domainname_new.append(0)
            domainname = domainname_new
            # Make up the maximum length of the domain name (0), or cut it out
            domain_len = len(domainname)
            if domain_len == 0:
                print(attr)
            line_mask = [1] * domain_len
            if domain_len < Data.max_domainname_len:
                padding_id = len(vocab)
                domainname.extend([padding_id] * (Data.max_domainname_len - domain_len))
                line_mask.extend([0] * (Data.max_domainname_len - domain_len))
            elif domain_len > Data.max_domainname_len:
                domainname = domainname[:Data.max_domainname_len]
            X.append(domainname)
            X_mask.append(line_mask)
            # Calculate the labels
            labels.append(attr.label)

        return X, X_mask, H, labels

    def createHyperEdgeByAttr(self, attrs: list):
        """
        Calculates the oversides that the property can construct.
        :param attr:
        :return:
        """
        # print(attr)
        # One edge per property
        rows = list()
        cols = list()
        values = list()
        cur_type_judge_index = 0
        while not attrs[cur_type_judge_index]:
            cur_type_judge_index += 1

            if cur_type_judge_index >= len(attrs):
                return []
        # print(cur_type_judge_index)
        # print(len(attr))
        # print("***************************")

        if isinstance(attrs[cur_type_judge_index], list):
            # Each property is a collection
            attr_edges = set()
            for item in attrs:
                if item is None:
                    continue
                for i in item:
                    attr_edges.add(i)

            for edge_index, edge in enumerate(attr_edges):
                # print(edge_index)
                # print(item)
                # print("*"*8)
                for node_index, node in enumerate(attrs):
                    if not node:
                        continue
                    if edge in node:
                        rows.append(edge_index)#Record the position of non-0 elements
                        cols.append(node_index)#Record the position of non-0 elements
                        values.append(1.0)
        else:
           # Each property is a collection

            attr_edges = set()
            for item in attrs:
                # print(item)
                if item is not None and item != "" :
                    if isinstance(item, list):
                        item = tuple(item)  # Convert the list to a tuple
                    attr_edges.add(item)

            for edge_index, edge in enumerate(attr_edges):
                for node_index, node in enumerate(attrs):
                    if node and node == edge:
                        rows.append(edge_index)
                        cols.append(node_index)
                        values.append(1.0)
        if rows:
            u_H = coo_matrix((values, (rows, cols)),
                             shape=(max(rows)+1, len(attrs)))  # shape=(len(attr), len(attr_edges))
            u_H = u_H.transpose().todense().A
            non_zero_hyp = u_H[:, np.sum(u_H, 0) > 1].tolist()
            #Return to the list non_zero_hyp
            return non_zero_hyp

        else:
            return []

    def createEdgeFromFeatureDT(self, nodenum, nodelist: list):
        res = list()
        for item in nodelist:
            tmplines = [0 for _ in range(nodenum)]
            for n in item:
                tmplines[n] = 1.0
            res.append(tmplines)
        return np.array(res).T#An array of edge information for a graph. Each column represents a node, each row represents a group, 1 indicates that the corresponding node appears in the group's decision path, and 0 indicates that it does not appear.
def divideToGroupByDT(sourcelist):#The decision path of the decision tree is divided into different groups, and a list containing the index of the nodes in the group is returned
    res = dict()
    for index in range(sourcelist.shape[0]):
        line = tuple(sourcelist[index].toarray()[0].tolist())#Converts the current row to a tuple
        if line not in res:
            res[line] = list()
        res[line].append(index)
    nodelist = list()
    for _, v in res.items():
        nodelist.append(v)
    return nodelist