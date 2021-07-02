# arguments - dataset(1) mode(random/all/normal)(2) model(dt/lr/nn)(3) cardinality(4) num_of_loops(5)
# save directory (6) #model_for_feature_keras (lstm/count/lemma) (7)


# python generic_generate_labels.py imdb normal dt 1 26 imdb_val2.5_sup5_dt1

import numpy as np
import matplotlib.pyplot as plt
import sys
from program_synthesis.heuristic_generator import HeuristicGenerator
from program_synthesis.synthesizer import Synthesizer
import pickle
import os
import warnings
from sklearn import model_selection as cross_validation
from lstm.DeepLSTM import *
warnings.filterwarnings("ignore")

dataset= sys.argv[1]
print('dataset is ', dataset)
loader_file = "data." + dataset+"_loader"

import importlib

load = importlib.import_module(loader_file)
feats = sys.argv[7]
dl = load.DataLoader()
train_primitive_matrix, val_primitive_matrix, test_primitive_matrix, train_ground,\
    val_ground, test_ground, vizer, val_idx, common_idx, train_text, val_text, test_text\
     = dl.load_data(dataset=dataset, split_val = 0.1, feat = feats)

x = [vizer.get_feature_names()[val_idx[i]] for i in common_idx ]

# val_primitive_matrix_heuristic, val_primitive_matrix, val_ground_heuristic,  val_ground= \
#         cross_validation.train_test_split(val_primitive_matrix, val_ground, test_size = 0.5, random_state=25)

# print('Size of validation heuristic set ', len(val_ground))
print('Size of validation set ', len(val_ground))
print('Size of train set ', len(train_ground))
print('Size of test set ', len(test_ground))

print('val_primitive_matrix.shape', val_primitive_matrix.shape)

num_classes = len(np.unique(train_ground))
model = sys.argv[3]
cardinality = int(sys.argv[4])
# hg = HeuristicGenerator(train_primitive_matrix, val_primitive_matrix,test_primitive_matrix,
#                         test_ground, val_ground, train_ground, b=0.5)
# hg.run_synthesizer(max_cardinality=1, idx=None, keep=3, model=model)

# syn = Synthesizer(val_primitive_matrix, val_ground, b=0.5)

# heuristics, feature_inputs = syn.generate_heuristics(model, 1)
# print ("Total Heuristics Generated: ", np.shape(heuristics)[1])


overall = {}
vals=[]
# total_lfs = int(np.shape(heuristics)[1])

mode=sys.argv[2]


save_dir = sys.argv[6]

save_path = "LFs/"+ dataset + "/" + save_dir
os.makedirs(save_path, exist_ok=True) 
# save_path = "generated_data/" + dataset #+ "/" + mode
print('save_path', save_path)
val_file_name = mode + '_val_LFs.npy'
train_file_name = mode + '_train_LFs.npy'
test_file_name = mode + '_test_LFs.npy'


if mode == 'all':
    keep_2nd = int(np.ceil(total_lfs/22))
    keep_1st = int(total_lfs % keep_2nd)
    print('keep_1st, keep_2nd', keep_1st, keep_2nd)
else:
    keep_1st=3
    keep_2nd=1

training_marginals = []
HF = []
num_loop = int(sys.argv[5])
for j in range(0,1):
    print('j',j)
    validation_accuracy = []
    training_accuracy = []
    validation_coverage = []
    training_coverage = []

    idx = None

    if j == 0:
        # print('value of j',j)
        hg = HeuristicGenerator(train_primitive_matrix, val_primitive_matrix, test_primitive_matrix,
                        test_ground, val_ground, train_ground, b=0.5)


    for i in range(3,num_loop):
        if (i-2)%5 == 0:
            print ("Running iteration: ", str(i-2))

        #Repeat synthesize-prune-verify at each iterations
        if i == 3:
            hg.run_synthesizer(max_cardinality= cardinality, idx=idx, keep=keep_1st, model=model, mode = mode)
        else:
            hg.run_synthesizer(max_cardinality= cardinality, idx=idx, keep=keep_2nd, model=model, mode = mode)
        hg.run_verifier()

        #Save evaluation metrics
        val_lfs, train_lfs = [], []
        hf = []
        va,ta, vc, tc, val_lfs, train_lfs, test_lfs, hf = hg.evaluate()
        HF = hf
        print(hf)
        break
        validation_accuracy.append(va)
        training_accuracy.append(ta)
        training_marginals.append(hg.vf.train_marginals)
        validation_coverage.append(vc)
        training_coverage.append(tc)
#         print('shape is ', val_lfs.shape, train_lfs.shape)

        if i==(num_loop-1):
    #         print('inside me ', labs[0])
            np.save(os.path.join(save_path ,val_file_name), val_lfs)
            np.save(os.path.join(save_path ,train_file_name), train_lfs)
            np.save(os.path.join(save_path ,test_file_name), test_lfs)
            print('labels saved') 
     
         
        #Find low confidence datapoints in the labeled set
        hg.find_feedback()
        idx = hg.feedback_idx
        print('Remaining to be labelled ', len(idx))

        #Stop the iterative process when no low confidence labels

        # if mode == 'normal' or mode == 'random':
        if idx == [] and j==0:
            np.save(os.path.join(save_path ,val_file_name), val_lfs)
            np.save(os.path.join(save_path ,train_file_name), train_lfs)
            np.save(os.path.join(save_path ,test_file_name), test_lfs)
            print('indexes exhausted... now saving labels')
            break
    vals.append(validation_accuracy[-1])
    print ("Program Synthesis Train Accuracy: ", training_accuracy[-1])
    print ("Program Synthesis Train Coverage: ", training_coverage[-1])
    print ("Program Synthesis Validation Accuracy: ", validation_accuracy[-1])


trx = np.load(os.path.join(save_path ,train_file_name))
valx = np.load(os.path.join(save_path ,val_file_name))
testx = np.load(os.path.join(save_path ,test_file_name))

def lsnork_to_l_m(lsnork, num_classes):
	m = 1 - np.equal(lsnork, -1).astype(int)
	l = m*lsnork + (1-m)*num_classes
	return l,m


# In[19]:

yoyo = list(range(1,num_classes))
yoyo.append(-1)
labels_lfs = []
idxs = []
for i in range(valx.shape[1]):
    for j in yoyo:
        if len(np.where(valx.T[i]==j)[0]) > 1:
            labels_lfs.append(j)
            idxs.append(i)
            break;
#         else:
#             labels_lfs.append(-1)

trx = trx[:,idxs]
testx = testx[:,idxs]
valx = valx[:,idxs]
print(trx.shape, valx.shape, testx.shape)
# print(testx)


lx = np.asarray(labels_lfs) #pahle me tha
lx[np.where(lx==-1)] = 0 #pahle me tha
print('LFS are ', lx)
file_name = mode + '_k.npy'
np.save(os.path.join(save_path , file_name), lx)
# exit()

if sys.argv[7] == 'lstm':
    mkt = MakeTokens()
    train_primitive_matrix, val_primitive_matrix, test_primitive_matrix, \
    vocab_size, embedding_vector_length, max_sentence_length =\
    mkt.make(train_text, val_text, test_text)

upto = int(len(val_ground)/2 ) # Size of d set i.e. Labelled set
d_L, U_L = val_ground[:upto], train_ground#[upto: ]
d_x, U_x = val_primitive_matrix[:upto], train_primitive_matrix#[upto: ] #features
d_l, U_l = valx[:upto,:], trx#[upto:,:] #LFs


d_d = np.array([1.0] * len(d_x))
d_r = np.zeros(d_l.shape) #rule exemplar coupling unavailable
d_L[np.where(d_L==-1)[0]] = 0

d_l[np.where(d_l==-1)]=10
d_l[np.where(d_l==0)]=-1
d_l[np.where(d_l==10)]=0
d_l, d_m = lsnork_to_l_m(d_l, num_classes)

save_dir = sys.argv[6]

pickle_save = "LFs/"+ dataset + "/" + save_dir


file_name = mode + '_d_processed.p'
with open(os.path.join(pickle_save, file_name),"wb") as f:
    pickle.dump(d_x,f)
    pickle.dump(d_l,f)
    pickle.dump(d_m,f)
    pickle.dump(d_L,f)
    pickle.dump(d_d,f)
    pickle.dump(d_r,f)



U_d = np.array([1.0] * len(U_x))
U_r = np.zeros(U_l.shape) #rule exemplar coupling unavailable

U_L[np.where(U_L==-1)[0]] = 0

U_l[np.where(U_l==-1)]=10
U_l[np.where(U_l==0)]=-1
U_l[np.where(U_l==10)]=0
U_l, U_m = lsnork_to_l_m(U_l, num_classes)
# U_L[np.where(U_L==-1)[0]] = 0

file_name = mode + '_U_processed.p'
with open(os.path.join(pickle_save, file_name),"wb") as f:
    pickle.dump(U_x,f)
    pickle.dump(U_l,f)
    pickle.dump(U_m,f)
    pickle.dump(U_L,f)
    pickle.dump(U_d,f)
    pickle.dump(U_r,f)



val_L = val_ground[upto:]
val_x = val_primitive_matrix[upto:] #features
val_l = valx[upto:,:] #L
val_d = np.array([1.0] * len(val_x))
val_r = np.zeros(val_l.shape) #rule exemplar coupling unavailable
val_L[np.where(val_L==-1)[0]] = 0

val_l[np.where(val_l==-1)]=10
val_l[np.where(val_l==0)]=-1
val_l[np.where(val_l==10)]=0
val_l, val_m = lsnork_to_l_m(val_l, num_classes)
file_name = mode + '_validation_processed.p'
with open(os.path.join(pickle_save,file_name),"wb") as f:
    pickle.dump(val_x,f)
    pickle.dump(val_l,f)
    pickle.dump(val_m,f)
    pickle.dump(val_L,f)
    pickle.dump(val_d,f)
    pickle.dump(val_r,f)


# In[33]:


test_L = test_ground
test_x = test_primitive_matrix #features
test_l = testx.copy() #L
test_d = np.array([1.0] * len(test_x))
test_r = np.zeros(test_l.shape) #rule exemplar coupling unavailable
test_L[np.where(test_L==-1)[0]] = 0

test_l[np.where(test_l==-1)]=10
test_l[np.where(test_l==0)]=-1
test_l[np.where(test_l==10)]=0

test_l, test_m = lsnork_to_l_m(test_l, num_classes)
file_name = mode + '_test_processed.p'

with open(os.path.join(pickle_save,file_name),"wb") as f:
    pickle.dump(test_x,f)
    pickle.dump(test_l,f)
    pickle.dump(test_m,f)
    pickle.dump(test_L,f)
    pickle.dump(test_d,f)
    pickle.dump(test_r,f)


print('Final Size of d set , U set  , validation set , test set', len(d_L), len(U_L), len(val_L), len(test_L))

# In[34]:


# objs = []
# with open('normal_validation_processed.p', 'rb') as f:
#     while 1:
#         try:
#             o = pickle.load(f)
#         except EOFError:
#             break
#         objs.append(o)


# # In[35]:


# import torch
# x_supervised = torch.tensor(objs[0]).double()
# y_supervised = torch.tensor(objs[3]).long()
# l_supervised = torch.tensor(objs[2]).long()
# s_supervised = torch.tensor(objs[2]).double()


# # In the plots above, we show the distribution of probabilistic labels Reef assigns to the training set in the first few iterations.
# # 
# # Next, we look at the accuracy and coverage of labels assigned to the training set in the _last_ iteration. The coverage is the percentage of training set datapoints that receive at least one label from the generated heuristics.

# # ### Save Training Set Labels 
# # We save the training set labels Reef generates that we use in the next notebook to train a simple LSTM model.

for i in hg.heuristic_stats().iloc[:len(idx)]['Feat 1']:
    print(x[int(i)])

filepath = './generated_data/' + dataset 
file_name = mode + '_reef.npy'
np.save(os.path.join(pickle_save,file_name), training_marginals[-1])
print('final LFs are ', lx.shape)
