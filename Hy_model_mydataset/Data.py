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
        # Characters that are not counted↓
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

    # Calculate information entropy
    for key in dataSet:
        prob = float(dataSet[key]) / numEntries  # Calculate p(xi)
        shannonEnt -= prob * log(prob, 2)  # log base 2
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
                                      ["id", "domainname", "ip4", "ip6", "cname", "ns", "mx", "soa_prins",
                                       "soa_respmail_addr", "soa_serial", "soa_refresh", "soa_retry", "soa_expire",
                                       "soa_ttl", "whois_tld", "whois_registrar", "whois_registrant_country",
                                       "whois_create_date", "whois_expire_date", "whois_last_update", "whois_status",
                                       "whois_statuses", "whois_dnssec", "whois_name_servers", "whois_registrant",
                                       "whois_email","tdl", "cdf","dian","label"])
    res = list()
    context = readDictFromJson(filepath)   # This is the complete original file in the library

    pattern = re.compile(r'\b91\.216\.248\.\d{1,3}\b')
   # Extracted domain name, this is the node feature of the classification
    for id, line in enumerate(context):
        domainname = line[1]
        ip4 = transDbFieldStrToList(line[2])
        ip6 = transDbFieldStrToList(line[3])
        if ip4 is not None:
            ip_list = [pattern.sub("null", ip) if pattern.match(ip) else ip for ip in ip4]
            ip4 = ip_list
            ip6="null"
        cname = transDbFieldStrToList(line[4])
        ns = transDbFieldStrToList(line[5])
        mx = transDbFieldStrToList(line[6])
        soa_prins = line[7]
        soa_resp_mail_addr = line[8]
        soa_serial = int(line[9]) if line[9] else 0
        soa_refresh = int(line[10]) if line[10] else 0
        soa_retry = int(line[11]) if line[11] else 0
        soa_expire = int(line[12]) if line[12] else 0
        soa_ttl = int(line[13]) if line[13] else 0
        whois_tld = line[14]
        whois_registrar = line[15]
        whois_registrant_country = line[16]
        whois_create_date = line[17]
        if whois_create_date is not None:
            create_date = datetime.datetime.strptime(whois_create_date, '%Y-%m-%d %H:%M:%S')
            time_delta = datetime.datetime.now() - create_date
            if time_delta.days < 365 * 2:  # If the time difference is less than two years
                whois_create_date = "short"
            else:
                whois_create_date = "long"
        whois_expire_date = line[18]
        whois_update_date = line[19]
        whois_status = line[20]
        whois_statuses = transDbFieldStrToList(line[21])
        whois_dnssec = line[22]
        whois_nameservers = transDbFieldStrToList(line[23])
        whois_registrant = line[24]
        whois_emails = line[25]
        tdl=extract_tld(domainname)
        shang=InfoEntropy(domainname)
        if shang>=3.5:
            cdf=1
        else:
            cdf=0
        dian=domainname.count(".")
        # label = line[26] if line[26] else "white"
        label = "malicious" if line[26] else "white"
        res.append(domainDataItemStruct(
            id=id,
            domainname=domainname,
            ip4=ip4,
            ip6=ip6,
            cname=cname,
            ns=ns,
            mx=mx,
            soa_prins=soa_prins,
            soa_respmail_addr=soa_resp_mail_addr,
            soa_serial=soa_serial,
            soa_refresh=soa_refresh,
            soa_retry=soa_retry,
            soa_expire=soa_expire,
            soa_ttl=soa_ttl,
            whois_tld=whois_tld,
            whois_registrar=whois_registrar,
            whois_registrant_country=whois_registrant_country,
            whois_create_date=whois_create_date,
            whois_expire_date=whois_expire_date,
            whois_last_update=whois_update_date,
            whois_status=whois_status,
            whois_statuses=whois_statuses,
            whois_dnssec=whois_dnssec,
            whois_name_servers=whois_nameservers,
            whois_registrant=whois_registrant,
            whois_email=whois_emails,
            tdl=tdl,
            cdf=cdf,
            dian=dian,
            label=label
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
            print("An exception has occurred")
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
            for c in attr.domainname:
                domainname_new.append(vocab[c])
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

            # Attributes are appended directly to the back
            attrs = [attr.ip4, attr.ip6, attr.cname, attr.ns, attr.mx, attr.soa_prins, attr.soa_respmail_addr,
                     attr.soa_serial, attr.soa_refresh, attr.soa_retry, attr.soa_expire, attr.soa_ttl, attr.whois_tld,
                     attr.whois_registrar, attr.whois_registrant_country, attr.whois_create_date,
                     attr.whois_expire_date, attr.whois_last_update, attr.whois_status, attr.whois_statuses,
                     attr.whois_dnssec, attr.whois_name_servers, attr.whois_registrant, attr.whois_email, attr.tdl, attr.cdf, attr.dian]

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
            line_tmp.append(item.soa_refresh if item.soa_refresh else 0)
            line_tmp.append(item.soa_retry if item.soa_retry else 0)
            line_tmp.append(item.soa_expire if item.soa_expire else 0)
            line_tmp.append(item.soa_ttl if item.soa_ttl else 0)
            line_tmp.append(1 if item.whois_registrar else 0)
            line_tmp.append(1 if item.whois_registrant_country else 0)
            line_tmp.append(2023 - int(item.whois_create_date.split("-")[0]) if item.whois_create_date else 0)
            line_tmp.append(int(item.whois_expire_date.split("-")[0]) if item.whois_expire_date else 0)
            line_tmp.append(1 if item.whois_last_update else 0)
            line_tmp.append(len(item.whois_status) if item.whois_status else 0)
            line_tmp.append(len(item.whois_statuses) if item.whois_statuses else 0)
            line_tmp.append(len(item.whois_dnssec) if item.whois_dnssec else 0)
            line_tmp.append(len(item.whois_name_servers) if item.whois_name_servers else 0)
            line_tmp.append(len(item.whois_registrant) if item.whois_registrant else 0)
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