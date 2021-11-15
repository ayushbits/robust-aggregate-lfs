#python generate_human_lfs.py dataset(imdb/trec/sms/youtube) count/lemma(2) savetype(dict/lemma) (3)
import random
import pickle
import re
import os
import numpy as np
import matplotlib.pyplot as plt
import sys
import warnings
from sklearn import model_selection as cross_validation
warnings.filterwarnings("ignore")

dataset= sys.argv[1]
print('dataset is ', dataset)
loader_file = "data." + dataset+"_loader"

import importlib

load = importlib.import_module(loader_file)
feats = sys.argv[2]
savetype = sys.argv[3]
# from obtain_embeddings import sentences_to_elmo_sentence_embs


# LABEL_DICT = {"ham": 1, "spam": 0}
LABEL_DICT = {"NUMERIC": 0, "LOCATION": 1, "HUMAN": 2, "DESCRIPTION": 3, "ENTITY": 4, "ABBREVIATION": 5}

def load_data(mode):
    label_map = {"DESC": "DESCRIPTION", "ENTY": "ENTITY", "HUM": "HUMAN", "ABBR": "ABBREVIATION", "LOC": "LOCATION",
           "NUM": "NUMERIC"}
    data = []

    with open(os.path.join("humanLFs", mode + '.txt', 'r'), encoding='latin1') as f:
        for line in f:
            label = LABEL_DICT[label_map[line.split()[0].split(":")[0]]]
            if mode == "test":
                sentence = (" ".join(line.split()[1:]))
            else:
                sentence = (" ".join(line.split(":")[1:])).lower().strip()
            data.append((sentence, label))
    return data


def load_rules():
    rules = []
    file_name = dataset+'_rule.txt'
    labels = []
    with open(os.path.join("humanLFs", file_name), 'r', encoding='latin1') as f:
        for line in f:
            list_in = line.strip().split("\t")
            label = LABEL_DICT[list_in[0]]
            pattern = list_in[1]
            sent = list_in[2].lower()
            rules.append((sent, label, pattern))
            labels.append(label)
            
    return rules, labels


class Generate_data:
    def __init__(self):
        self.num_labels = len(LABEL_DICT)
        # self.train_data = load_data("train")
        # self.validation_data = load_data("valid")
        # self.test_data = load_data("test")
        self.rules, self.labels = load_rules()
        self.num_rules = len(self.rules)
        dl = load.DataLoader()
        self.train_feats, self.val_feats, self.test_feats, self.train_ground, self.val_ground,\
        self.test_ground, _,_,_, self.train_data, self.validation_data,\
        self.test_data = dl.load_data(dataset=dataset, split_val = 0.1, feat = feats)
        self.save_dir = "humanLFs/"+ dataset + "/" + savetype
        np.save(os.path.join(self.save_dir, 'k.npy'), self.labels)



    def fire_rules(self,sentence):
        #returns m and l values for the sentence
        m = np.zeros(self.num_rules)
        l = self.num_labels + np.zeros(self.num_rules)

        for rid,(sent,lab,pat) in enumerate(self.rules):
            pattern = re.compile(pat)
            rule_label = lab
            result = re.findall(pattern, sentence.strip().lower())                        
            if result:
                m[rid] = 1
                l[rid] = lab
        return m,l


    def _geneate_pickles(self,mode):
        if mode == "U":
            data = self.train_data
        elif mode == "test":
            data = self.test_data
        elif mode == "validation":
            data = self.validation_data
        else:
            print("Error: Wrong mode")
            exit()

        xx = []
        xl = []
        xm = []
        xL = []
        xd = []
        xr = []

        # for sentence in data:
        for i, sentence in enumerate(data):
            # xx.append(sentence)
            if mode == 'U':
                # print('shape hai ', self.train_feats.shape)
                xx = self.train_feats
            elif mode == 'validation':
                xx = self.val_feats
            elif mode == 'test':
                xx = self.test_feats
            if mode == "U":
                xL.append(self.num_labels)
            else:
                # xL.append(label)
                if mode == 'validation':
                    xL.append(self.val_ground[i])
                elif mode == 'test':
                    xL.append(self.test_ground[i])

            xd.append(0)
            xr.append(np.zeros(self.num_rules))
            m,l = self.fire_rules(sentence)
            xm.append(m)
            xl.append(l)

        if mode == "validation":
            length = int(len(xL)/2)
            d_len = int(length*1.5)
            print('Length is', d_len)
            print(np.array(xx).shape)
            print(np.array(xl).shape)
            print(np.array(xm).shape)
            print(np.unique(xL))
            print(np.array(xd).shape)
            print(np.array(xr).shape)

            with open(os.path.join(self.save_dir,"d_processed.p"), "wb") as pkl_f:
                # for sentence in d_x:
                #     txt_f.write(sentence.strip()+'\n')
                # print(len(d_x))
                # d_x = sentences_to_elmo_sentence_embs(d_x)
                # print(len(d_x))
                pickle.dump(np.array(xx[0:d_len,:]), pkl_f)
                pickle.dump(np.array(xl[0:d_len]), pkl_f)
                pickle.dump(np.array(xm[0:d_len]), pkl_f)
                pickle.dump(np.array(xL[0:d_len]), pkl_f)
                pickle.dump(np.array(xd[0:d_len]), pkl_f)
                pickle.dump(np.array(xr[0:d_len]), pkl_f)

            with open(os.path.join(self.save_dir,"validation_processed.p"), "wb") as pkl_f:

                # for sentence in d_x:
                #     txt_f.write(sentence.strip()+'\n')
                # print(len(d_x))
                # d_x = sentences_to_elmo_sentence_embs(d_x)
                # print(len(d_x))
                pickle.dump(np.array(xx[d_len:]), pkl_f)
                pickle.dump(np.array(xl[d_len:]), pkl_f)
                pickle.dump(np.array(xm[d_len:]), pkl_f)
                pickle.dump(np.array(xL[d_len:]), pkl_f)
                pickle.dump(np.array(xd[d_len:]), pkl_f)
                pickle.dump(np.array(xr[d_len:]), pkl_f)


        else:

            with open(os.path.join(self.save_dir,"{}_processed.p").format(mode),"wb") as pkl_f:
                # for sentence in xx:
                #     txt_f.write(sentence.strip()+'\n')
                # print(len(xx))
                # xx = sentences_to_elmo_sentence_embs(xx,128)
                

                print('Length of xx ', len(xx))
                print(np.array(xx).shape)
                print(np.array(xl).shape)
                print(np.array(xm).shape)
                print(np.unique(xL))
                print(np.array(xd).shape)
                print(np.array(xr).shape)
                pickle.dump(np.array(xx),pkl_f)
                pickle.dump(np.array(xl),pkl_f)
                pickle.dump(np.array(xm),pkl_f)
                pickle.dump(np.array(xL),pkl_f)
                pickle.dump(np.array(xd),pkl_f)
                pickle.dump(np.array(xr),pkl_f)

    def generate_pickles(self):
        #=== d_processed.p ====#
        d_x = []
        d_l = []
        d_m = []
        d_L = []
        d_d = []
        d_r = []
        # for rule_id,(sentence,label,pattern) in enumerate(self.rules):
        #     if sentence in d_x:
        #         s_idx = d_x.index(sentence)
        #         if label == d_L[s_idx]:
        #             d_r[s_idx][rule_id]=1
        #             continue
        #     d_x.append(sentence)
        #     d_d.append(1)
        #     d_L.append(label)
        #     m = np.zeros(self.num_rules)
        #     l = self.num_labels + np.zeros(self.num_rules)
        #     r = np.zeros(self.num_rules)
        #     r[rule_id] = 1
        #     d_r.append(r)
        #     m,l = self.fire_rules(sentence)
        #     assert m[rule_id] == 1
        #     d_m.append(m)
        #     d_l.append(l)
        # with open("d_processed.p", "wb") as pkl_f, open("d_sentences.txt","w") as txt_f:
        #     for sentence in d_x:
        #         txt_f.write(sentence.strip()+'\n')
        #     print(len(d_x))
        #     d_x = sentences_to_elmo_sentence_embs(d_x)
        #     print(len(d_x))
        #     pickle.dump(np.array(d_x), pkl_f)
        #     pickle.dump(np.array(d_l), pkl_f)
        #     pickle.dump(np.array(d_m), pkl_f)
        #     pickle.dump(np.array(d_L), pkl_f)
        #     pickle.dump(np.array(d_d), pkl_f)
        #     pickle.dump(np.array(d_r), pkl_f)
        #====== U_processed ======#
        self._geneate_pickles("test")
        self._geneate_pickles("validation")
        self._geneate_pickles("U")
        



if __name__ == '__main__':
    obj = Generate_data()
    obj.generate_pickles()
