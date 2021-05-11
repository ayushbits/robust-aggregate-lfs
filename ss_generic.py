import torch
import sys
import numpy as np
from logistic_regression import *
from deep_net import *
import warnings
warnings.filterwarnings("ignore")
from cage import *
from sklearn.feature_extraction.text import TfidfVectorizer
from losses import *
import pickle
from torch.utils.data import TensorDataset, DataLoader
import wandb
wandb.init(project='generic', entity='spear-plus')
conf = wandb.config
# CUDA for PyTorch
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
torch.backends.cudnn.benchmark = True

torch.set_default_dtype(torch.float64)
torch.set_printoptions(threshold=20)

objs = []
dset_directory = sys.argv[10]
n_classes = int(sys.argv[11])
feat_model = sys.argv[12]
qg_available = int(sys.argv[13])
batch_size = int(sys.argv[14])
lr_fnetwork = float(sys.argv[15])
lr_gm = float(sys.argv[16])
name_dset = dset_directory.split("/")[-1].lower()
print('dset is ', name_dset)
mode = sys.argv[17] #''
metric = sys.argv[18]
conf.learning_rate = lr_fnetwork #wandb
wrunname = name_dset + "_" + mode +"_generic"#wandb
wandb.run.name = wrunname #wandb


from sklearn.metrics import precision_score as prec_score
from sklearn.metrics import recall_score as recall_score
if metric=='accuracy':
    from sklearn.metrics import accuracy_score as score
    print('inside accuracy')
else:
    from sklearn.metrics import f1_score as score
    metric_avg = 'macro'


if mode != '':
    fname = dset_directory + "/" + mode + "_d_processed.p"
else:
    fname = dset_directory + "/d_processed.p"
with open(fname, 'rb') as f:
    while 1:
        try:
            o = pickle.load(f)
        except EOFError:
            break
        objs.append(o)

x_supervised = torch.tensor(objs[0]).double()
y_supervised = torch.tensor(objs[3]).long()
l_supervised = torch.tensor(objs[2]).long()
s_supervised = torch.tensor(objs[2]).double()
print('supervised shape', objs[2].shape)

objs = []
if mode != '':
    fname = dset_directory + "/" + mode + "_U_processed.p"
else:
    fname = dset_directory + "/U_processed.p"

with open(fname, 'rb') as f:
    while 1:
        try:
            o = pickle.load(f)
        except EOFError:
            break
        objs.append(o)

excl= []
idx=0
for x in objs[1]:
    if(all(x==int(n_classes))):
        excl.append(idx)
    idx+=1
print('no of excludings are ', len(excl))

x_unsupervised = torch.tensor(np.delete(objs[0],excl, axis=0)).double()
y_unsupervised = torch.tensor(np.delete(objs[3],excl, axis=0)).long()
l_unsupervised = torch.tensor(np.delete(objs[2],excl, axis=0)).long()
s_unsupervised = torch.tensor(np.delete(objs[2],excl, axis=0)).double()
print('UNsupervised shape', objs[2].shape)
print('Length of U is', len(x_unsupervised))

objs = []
if mode != '':
    fname = dset_directory + "/" + mode + "_validation_processed.p"
else:
    fname = dset_directory + "/validation_processed.p"

with open(fname, 'rb') as f:
    while 1:
        try:
            o = pickle.load(f)
        except EOFError:
            break
        objs.append(o)

x_valid = torch.tensor(objs[0]).double()
y_valid = objs[3]
l_valid = torch.tensor(objs[2]).long()
s_valid = torch.tensor(objs[2]).double()
print('Valid shape', objs[2].shape)
objs1 = []
if mode != '':
    fname = dset_directory + "/" + mode + "_test_processed.p"
else:
    fname = dset_directory + "/test_processed.p"


with open(fname, 'rb') as f:
    while 1:
        try:
            o = pickle.load(f)
        except EOFError:
            break
        objs1.append(o)
x_test = torch.tensor(objs1[0]).double()
y_test = objs1[3]
l_test = torch.tensor(objs1[2]).long()
s_test = torch.tensor(objs1[2]).double()
print('Test shape', objs1[2].shape)

n_features = x_supervised.shape[1]

# Labeling Function Classes
# k = torch.from_numpy(np.array([0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0])).long()
#lf_classes_file = sys.argv[11]

if mode != '':
    fname = dset_directory + '/' + mode + '_k.npy'
else:
    fname = dset_directory + '/k.npy'
k = torch.from_numpy(np.load(fname)).long()
n_lfs = len(k)
print('LFs are ',k)
print('no of lfs are ', n_lfs)

# a = torch.ones(n_lfs).double() * 0.9
# print('before ',a)
if qg_available:
    a = torch.from_numpy(np.load(dset_directory+'/prec.npy')).double()
else:
    # a = torch.ones(n_lfs).double() * 0.9

    prec_lfs=[]
    for i in range(n_lfs):
       correct = 0
       for j in range(len(y_valid)):
           if y_valid[j] == l_valid[j][i]:
               correct+=1
       prec_lfs.append(correct/len(y_valid))
    a = torch.tensor(prec_lfs).double()

# n_lfs = int(len(k))
# print('number of lfs ', n_lfs)
# a = torch.ones(n_lfs).double() * 0.9
continuous_mask = torch.zeros(n_lfs).double()


for i in range(s_supervised.shape[0]):
    for j in range(s_supervised.shape[1]):
        if s_supervised[i, j].item() > 0.999:
            s_supervised[i, j] = 0.999
        if s_supervised[i, j].item() < 0.001:
            s_supervised[i, j] = 0.001

for i in range(s_unsupervised.shape[0]):
    for j in range(s_unsupervised.shape[1]):
        if s_unsupervised[i, j].item() > 0.999:
            s_unsupervised[i, j] = 0.999
        if s_unsupervised[i, j].item() < 0.001:
            s_unsupervised[i, j] = 0.001

for i in range(s_valid.shape[0]):
    for j in range(s_valid.shape[1]):
        if s_valid[i, j].item() > 0.999:
            s_valid[i, j] = 0.999
        if s_valid[i, j].item() < 0.001:
            s_valid[i, j] = 0.001

for i in range(s_test.shape[0]):
    for j in range(s_test.shape[1]):
        if s_test[i, j].item() > 0.999:
            s_test[i, j] = 0.999
        if s_test[i, j].item() < 0.001:
            s_test[i, j] = 0.001



l = torch.cat([l_supervised, l_unsupervised])
s = torch.cat([s_supervised, s_unsupervised])
x_train = torch.cat([x_supervised, x_unsupervised])
y_train = torch.cat([y_supervised, y_unsupervised])
supervised_mask = torch.cat([torch.ones(l_supervised.shape[0]), torch.zeros(l_unsupervised.shape[0])])
print('X_train', x_train.shape, 'l',l.shape, 's', s.shape)

## Quality Guides ##



## End Quality Quides##
# a =  torch.tensor(np.load(dset_directory + '/precision_values.npy'))
# print('after ',a)

#Setting |validation|=|supevised|
# x_valid = x_valid[0:len(x_supervised)]
# y_valid = y_valid[0:len(x_supervised)]
# s_valid = s_valid[0:len(x_supervised)]
# l_valid = l_valid[0:len(x_supervised)]

# print(l_valid.shape)
# print(l_valid[0])

num_runs = int(sys.argv[9])

final_score_gm, final_score_lr, final_score_gm_val, final_score_lr_val = [],[],[],[]

final_score_lr_prec, final_score_lr_recall, final_score_gm_prec, final_score_gm_recall = [],[],[],[]
for lo in range(0,num_runs):
    pi = torch.ones((n_classes, n_lfs)).double()
    pi.requires_grad = True

    theta = torch.ones((n_classes, n_lfs)).double() * 1
    theta.requires_grad = True

    pi_y = torch.ones(n_classes).double()
    pi_y.requires_grad = True

    if feat_model == 'lr':
        lr_model = LogisticRegression(n_features, n_classes)
    elif feat_model =='nn':
        n_hidden = 512
        lr_model = DeepNet(n_features, n_hidden, n_classes)
    else:
        print('Please provide feature based model : lr or nn')
        exit()


    optimizer = torch.optim.Adam([{"params": lr_model.parameters()}, {"params": [pi, pi_y, theta]}], lr=0.001)
    optimizer_lr = torch.optim.Adam(lr_model.parameters(), lr=lr_fnetwork)
    optimizer_gm = torch.optim.Adam([theta, pi, pi_y], lr=lr_gm, weight_decay=0)
    # optimizer = torch.optim.Adam([theta, pi, pi_y], lr=0.01, weight_decay=0)
    supervised_criterion = torch.nn.CrossEntropyLoss()



    dataset = TensorDataset(x_train, y_train, l, s, supervised_mask)

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,pin_memory=True)
    save_folder = sys.argv[1]
    print('num runs are ', sys.argv[1], num_runs)
    best_score_lr,best_score_gm,best_epoch_lr,best_epoch_gm,best_score_lr_val, best_score_gm_val = 0,0,0,0,0,0
    best_score_lr_prec,best_score_lr_recall ,best_score_gm_prec,best_score_gm_recall= 0,0,0,0

    stop_pahle, stop_pahle_gm = [], []
    # wandb.watch(lr_model)
    for epoch in range(100):
        lr_model.train()

        for batch_ndx, sample in enumerate(loader):
            optimizer_lr.zero_grad()
            optimizer_gm.zero_grad()

            unsup = []
            sup = []
            supervised_indices = sample[4].nonzero().view(-1)
            # unsupervised_indices = indices  ## Uncomment for entropy
            unsupervised_indices = (1-sample[4]).nonzero().squeeze().view(-1)
            # print('sample[2][unsupervised_indices].shape', sample[2][unsupervised_indices].shape)
            # print('sample[3][unsupervised_indices].shape', sample[3][unsupervised_indices].shape)


            if(sys.argv[2] =='l1'):
                if len(supervised_indices) > 0:
                    loss_1 = supervised_criterion(lr_model(sample[0][supervised_indices]), sample[1][supervised_indices])
                else:
                    loss_1 = 0
            else:
                loss_1=0

            if(sys.argv[3] =='l2'):
                unsupervised_lr_probability = torch.nn.Softmax()(lr_model(sample[0][unsupervised_indices]))
                loss_2 = entropy(unsupervised_lr_probability)
            else:
                loss_2=0
            if(sys.argv[4] =='l3'):
                # print(theta)
                y_pred_unsupervised = np.argmax(probability(theta, pi_y, pi, sample[2][unsupervised_indices], sample[3][unsupervised_indices], k, n_classes,continuous_mask).detach().numpy(), 1)
                loss_3 = supervised_criterion(lr_model(sample[0][unsupervised_indices]), torch.tensor(y_pred_unsupervised))
            else:
                loss_3 = 0

            if (sys.argv[5] == 'l4' and len(supervised_indices) > 0):
                loss_4 = log_likelihood_loss_supervised(theta, pi_y, pi, sample[1][supervised_indices], sample[2][supervised_indices], sample[3][supervised_indices], k, n_classes, continuous_mask)
            else:
                loss_4 = 0

            if(sys.argv[6] =='l5'):
                loss_5 = log_likelihood_loss(theta, pi_y, pi, sample[2][unsupervised_indices], sample[3][unsupervised_indices], k, n_classes, continuous_mask)
            else:
                loss_5 =0

            if(sys.argv[7] =='l6'):
                if(len(supervised_indices) >0):
                    supervised_indices = supervised_indices.tolist()
                    probs_graphical = probability(theta, pi_y, pi, torch.cat([sample[2][unsupervised_indices], sample[2][supervised_indices]]),\
                    torch.cat([sample[3][unsupervised_indices],sample[3][supervised_indices]]), k, n_classes, continuous_mask)
                else:
                    probs_graphical = probability(theta, pi_y, pi,sample[2][unsupervised_indices],sample[3][unsupervised_indices],\
                         k, n_classes, continuous_mask)
                probs_graphical = (probs_graphical.t() / probs_graphical.sum(1)).t()
                probs_lr = torch.nn.Softmax()(lr_model(sample[0]))
                loss_6 = kl_divergence(probs_lr, probs_graphical)
                # loss_6 = kl_divergence(probs_graphical, probs_lr) #original version

            else:
                loss_6= 0
            # loss_6 = - torch.log(1 - probs_graphical * (1 - probs_lr)).sum(1).mean()
            if(sys.argv[8] =='qg'):
                prec_loss = precision_loss(theta, k, n_classes, a)
            else:
                prec_loss =0

            loss = loss_1 + loss_2 + loss_3 + loss_4 + loss_6+loss_5 + prec_loss
#            print('loss is',loss_1, loss_2, loss_3, loss_4, loss_5, loss_6, prec_loss)
            if loss != 0:
                loss.backward()
                optimizer_gm.step()
                optimizer_lr.step()
        wname = "Run_"+str(lo)+" Train Loss" #wandb
        wandb.log({wname:loss, 'custom_step':epoch}) #wandb
        y_pred = np.argmax(probability(theta, pi_y, pi, l_test, s_test, k, n_classes, continuous_mask).detach().numpy(), 1)
        if metric=='accuracy':
            gm_acc = score(y_test, y_pred)
            lr_prec = prec_score(y_test, y_pred, average=None) 
            lr_recall = recall_score(y_test, y_pred, average=None)
            gm_prec, gm_recall = 0,0
        else:
            gm_acc = score(y_test, y_pred, average=metric_avg)
            gm_prec = prec_score(y_test, y_pred, average=metric_avg)
            gm_recall = recall_score(y_test, y_pred, average=metric_avg)
        #Valid
        y_pred = np.argmax(probability(theta, pi_y, pi, l_valid, s_valid, k, n_classes, continuous_mask).detach().numpy(), 1)
        
        if metric=='accuracy':
            gm_valid_acc = score(y_valid, y_pred)
            gm_prec, gm_recall = 0,0
        else:
            gm_valid_acc = score(y_valid, y_pred, average=metric_avg)

        #LR Test

        probs = torch.nn.Softmax()(lr_model(x_test))
        y_pred = np.argmax(probs.detach().numpy(), 1)
        # if name_dset =='youtube' or name_dset=='census' or name_dset =='sms':
        if metric=='accuracy':
        	# print('inside accuracy LR test')
        	lr_acc =score(y_test, y_pred)
        	lr_prec = prec_score(y_test, y_pred, average=None)
        	lr_recall = recall_score(y_test, y_pred, average=None)
        	gm_prec, gm_recall = 0,0

    
        else:
            lr_acc =score(y_test, y_pred, average=metric_avg)
            lr_prec = prec_score(y_test, y_pred, average=metric_avg)
            lr_recall = recall_score(y_test, y_pred, average=metric_avg)
        #LR Valid
        probs = torch.nn.Softmax()(lr_model(x_valid))
        y_pred = np.argmax(probs.detach().numpy(), 1)
        
        if metric=='accuracy':
            lr_valid_acc = score(y_valid, y_pred)
            gm_prec, gm_recall = 0,0
        else:
            lr_valid_acc = score(y_valid, y_pred, average=metric_avg)
        print("Epoch: {}\t Test GM accuracy_score: {}".format(epoch, gm_acc ))
#         print("Epoch: {}\tGM accuracy_score(Valid): {}".format(epoch, gm_valid_acc))
        print("Epoch: {}\tTest LR accuracy_score: {}".format(epoch, lr_acc))    
#         print("Epoch: {}\tLR accuracy_score(Valid): {}".format(epoch, lr_valid_acc))
        wname = "Run_"+str(lo)+" LR valid score"
        wnamegm = 'Run_' + str(lo) + ' GM valid score'
        wandb.log({wname:lr_valid_acc, 
            wnamegm:gm_valid_acc,'custom_step':epoch})

        if epoch > 1 and gm_valid_acc >= best_score_gm_val and gm_valid_acc >= best_score_lr_val:
            # print("Inside Best hu Epoch: {}\t Test GM accuracy_score: {}".format(epoch, gm_acc ))
            # print("Inside Best hu Epoch: {}\tGM accuracy_score(Valid): {}".format(epoch, gm_valid_acc))
            if gm_valid_acc == best_score_gm_val or gm_valid_acc == best_score_lr_val:
                if best_score_gm < gm_acc or best_score_lr < lr_acc:
                    best_epoch_lr = epoch
                    best_score_lr_val = lr_valid_acc
                    best_score_lr = lr_acc

                    best_epoch_gm = epoch
                    best_score_gm_val = gm_valid_acc
                    best_score_gm = gm_acc

                    best_score_lr_prec = lr_prec
                    best_score_lr_recall  = lr_recall
                    best_score_gm_prec = gm_prec
                    best_score_gm_recall  = gm_recall
            else:
                best_epoch_lr = epoch
                best_score_lr_val = lr_valid_acc
                best_score_lr = lr_acc

                best_epoch_gm = epoch
                best_score_gm_val = gm_valid_acc
                best_score_gm = gm_acc

                best_score_lr_prec = lr_prec
                best_score_lr_recall  = lr_recall
                best_score_gm_prec = gm_prec
                best_score_gm_recall  = gm_recall
                stop_pahle = []
                stop_pahle_gm = []
            checkpoint = {'theta': theta,'pi': pi}
            # torch.save(checkpoint, save_folder+"/gm_"+str(epoch)    +".pt")
            checkpoint = {'params': lr_model.state_dict()}
            # torch.save(checkpoint, save_folder+"/lr_"+ str(epoch)+".pt")
            

        if epoch > 1 and lr_valid_acc >= best_score_lr_val and lr_valid_acc >= best_score_gm_val:
            # print("Inside Best hu Epoch: {}\tTest LR accuracy_score: {}".format(epoch, lr_acc ))
            # print("Inside Best hu Epoch: {}\tLR accuracy_score(Valid): {}".format(epoch, lr_valid_acc))
            if lr_valid_acc == best_score_lr_val or lr_valid_acc == best_score_gm_val:
                if best_score_lr < lr_acc or best_score_gm < gm_acc:
                    
                    best_epoch_lr = epoch
                    best_score_lr_val = lr_valid_acc
                    best_score_lr = lr_acc

                    best_epoch_gm = epoch
                    best_score_gm_val = gm_valid_acc
                    best_score_gm = gm_acc

                    best_score_lr_prec = lr_prec
                    best_score_lr_recall  = lr_recall
                    best_score_gm_prec = gm_prec
                    best_score_gm_recall  = gm_recall
            else:
                best_epoch_lr = epoch
                best_score_lr_val = lr_valid_acc
                best_score_lr = lr_acc
                best_epoch_gm = epoch
                best_score_gm_val = gm_valid_acc
                best_score_gm = gm_acc
                best_score_lr_prec = lr_prec
                best_score_lr_recall  = lr_recall
                best_score_gm_prec = gm_prec
                best_score_gm_recall  = gm_recall
                stop_pahle = []
                stop_pahle_gm = []
            checkpoint = {'theta': theta,'pi': pi}
            # torch.save(checkpoint, save_folder+"/gm_"+str(epoch)    +".pt")
            checkpoint = {'params': lr_model.state_dict()}
            # torch.save(checkpoint, save_folder+"/lr_"+ str(epoch)+".pt")
            


        # if len(stop_pahle) > 20 and len(stop_pahle_gm) > 20 and (all(best_score_lr_val >= k for k in stop_pahle) and all(best_score_gm_val >= k for k in stop_pahle_gm)):
        if len(stop_pahle) > 10 and len(stop_pahle_gm) > 10 and (all(best_score_lr_val >= k for k in stop_pahle)):
    #    if  len(stop_pahle_gm) > 10 and all(best_score_gm_val >= k for k in stop_pahle_gm):
        
            print('Early Stopping at', best_epoch_gm, best_score_gm, best_score_lr)
            print('Validation score Early Stopping at', best_epoch_gm, best_score_lr_val, best_score_gm_val)
            break
        else:
            # print('inside else stop pahle epoch', epoch)
            stop_pahle.append(lr_valid_acc)
            stop_pahle_gm.append(gm_valid_acc)

    # print("Run \t",lo, "Epoch Gm, Epoch LR, GM, LR \t", best_epoch_gm, best_epoch_lr,best_score_gm, best_score_lr)
    # print("Run \t",lo, "GM Val, LR Val \t", best_score_gm_val, best_score_lr_val)
    print('Best Epoch LR', best_epoch_lr)
    print('Best Precision LR', best_score_lr_prec)
    print('Best Recall LR', best_score_lr_recall)
    print('Best Epoch GM', best_epoch_gm)
    print('Best Precision GM ', best_score_gm_prec)
    print('Best Recall GM ', best_score_gm_recall)
    print("Run \t",lo, "Epoch, GM, LR \t", best_score_gm, best_score_lr)
    print("Run \t",lo, "GM Val, LR Val \t", best_score_gm_val, best_score_lr_val)
    final_score_gm.append(best_score_gm)
    final_score_lr.append(best_score_lr)
#    final_score_lr_prec.append(best_score_lr_prec)
#    final_score_lr_recall.append(best_score_lr_recall)

#    final_score_gm_prec.append(best_score_gm_prec)
#    final_score_gm_recall.append(best_score_gm_recall)

    final_score_gm_val.append(best_score_gm_val)
    final_score_lr_val.append(best_score_lr_val)


print("===================================================")
print("TEST Averaged scores are for LR", np.mean(final_score_lr))
print("TEST Precision average scores are for LR", np.mean(best_score_lr_prec))
print("TEST Recall average scores are for LR", np.mean(best_score_lr_recall))
print("===================================================")
print("TEST Averaged scores are for GM",  np.mean(final_score_gm))
#print("TEST Precision average scores are for GM", np.mean(final_score_gm_prec))
#print("TEST Recall average scores are for GM", np.mean(final_score_gm_recall))
print("===================================================")
print("VALIDATION Averaged scores are for GM,LR", np.mean(final_score_gm_val), np.mean(final_score_lr_val))
print("TEST STD  are for GM,LR", np.std(final_score_gm), np.std(final_score_lr))
print("VALIDATION STD  are for GM,LR", np.std(final_score_gm_val), np.std(final_score_lr_val))

wandb.log({'test_lr':np.mean(final_score_lr),'test_gm':np.mean(final_score_gm)})#wandb
wandb.log({'test_mean_GM ':np.mean(final_score_lr), 'test_mean_GM': np.mean(final_score_gm)}) #wandb
wandb.log({'test_STD_LR ':np.std(final_score_lr), 'test_STD_GM': np.std(final_score_gm)}) #wandb
