import torch
import sys
import numpy as np
from logistic_regression import *
from deep_net import *
import warnings
warnings.filterwarnings("ignore")
from argparse import ArgumentParser
# from cage import *
from sklearn.feature_extraction.text import TfidfVectorizer
from losses import *
import pickle
from torch.utils.data import TensorDataset, DataLoader
import os
import wandb
wandb.init(project='rewt', entity='spear-plus')
conf = wandb.config
# CUDA for PyTorch
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
print(device)
torch.backends.cudnn.benchmark = True

torch.set_default_dtype(torch.float64)
torch.set_printoptions(threshold=20)

print('sys.argv[1]', sys.argv[1])
dset_directory = sys.argv[10]
n_classes = int(sys.argv[11])
feat_model = sys.argv[12]
qg_available = int(sys.argv[13])
batch_size = int(sys.argv[14])
lr_fnetwork = float(sys.argv[15])
lr_gm = float(sys.argv[16])
name_dset = dset_directory.split("/")[-1].lower()
print('dset is ', name_dset)
conf.learning_rate = lr_fnetwork #wandb
mode = sys.argv[17]
metric = sys.argv[18]

objs = []
wrunname = name_dset + "_" + mode +"_reweight"#wandb
wandb.run.name = wrunname #wandb
# if name_dset =='youtube' or name_dset=='census':
if metric=='accuracy':
    from sklearn.metrics import accuracy_score as score
    print('inside accuracy')
else:
    from sklearn.metrics import f1_score as score
    from sklearn.metrics import precision_score as prec_score
    from sklearn.metrics import recall_score as recall_score
    metric_avg = 'macro'


from gpu_weighted_cage import *
import higher
import copy

lam1 = 1
def rewt_lfs(sample, lr_model, theta, pi_y, pi, wts):
    wts_param = torch.nn.Parameter(wts, requires_grad=True)
    lr_model.register_parameter("wts", wts_param)
    theta_param = torch.nn.Parameter(theta, requires_grad=True)
    lr_model.register_parameter("theta", theta_param)
    pi_y_param = torch.nn.Parameter(pi_y, requires_grad=True)
    lr_model.register_parameter("pi_y", pi_y_param)
    pi_param = torch.nn.Parameter(pi, requires_grad=True)
    lr_model.register_parameter("pi", pi_param)
    # print(lr_model.linear.weight)
    if feat_model == 'lr':
        optimizer = torch.optim.Adam([
                {'params': lr_model.linear.weight},# linear_1.parameters()},
                 # {'params': lr_model['params']['linear.bias']},
                # {'params':lr_model.linear_2.parameters()},
                # {'params':lr_model.out.parameters()},
                {'params': [lr_model.theta, lr_model.pi, lr_model.pi_y], 'lr': 0.01, 'weight_decay':0}
            ], lr=1e-4)
    elif feat_model =='nn':
        optimizer = torch.optim.Adam([
                {'params': lr_model.linear_1.parameters()},
                {'params':lr_model.linear_2.parameters()},
                {'params':lr_model.out.parameters()},
                {'params': [lr_model.theta, lr_model.pi, lr_model.pi_y], 'lr': 0.01, 'weight_decay':0}
            ], lr=1e-4)
    with higher.innerloop_ctx(lr_model, optimizer) as (fmodel, diffopt):
        supervised_criterion = torch.nn.CrossEntropyLoss()
        optimizer.zero_grad()
        supervised_indices = sample[4].nonzero().view(-1)
        unsupervised_indices = (1 - sample[4]).nonzero().squeeze()
        if len(supervised_indices) > 0:
            loss_1 = supervised_criterion(fmodel(sample[0][supervised_indices]), sample[1][supervised_indices])
        else:
            loss_1 = 0
        unsupervised_lr_probability = torch.nn.Softmax(dim=1)(fmodel(sample[0][unsupervised_indices]).view(-1, n_classes))
        loss_2 = entropy(unsupervised_lr_probability)
        y_pred_unsupervised = np.argmax(
            probability(fmodel.theta, fmodel.pi_y, fmodel.pi, sample[2][unsupervised_indices], sample[3][unsupervised_indices], k, n_classes,
                        continuous_mask, fmodel.wts).detach().numpy(), 1)
        loss_3 = supervised_criterion(fmodel(sample[0][unsupervised_indices]), torch.tensor(y_pred_unsupervised))
        if len(supervised_indices) > 0:
            loss_4 = log_likelihood_loss_supervised(fmodel.theta, fmodel.pi_y, fmodel.pi, sample[1][supervised_indices],
                                                    sample[2][supervised_indices], sample[3][supervised_indices], k,
                                                    n_classes,
                                                    continuous_mask, fmodel.wts)
        else:
            loss_4 = 0
        loss_5 = log_likelihood_loss(fmodel.theta, fmodel.pi_y, fmodel.pi, sample[2][unsupervised_indices], sample[3][unsupervised_indices],
                                     k, n_classes, continuous_mask, fmodel.wts)
        prec_loss = precision_loss(fmodel.theta, k, n_classes, a, fmodel.wts)
        probs_graphical = probability(fmodel.theta, fmodel.pi_y, fmodel.pi, sample[2], sample[3], k, n_classes, continuous_mask, fmodel.wts)
        probs_graphical = (probs_graphical.t() / probs_graphical.sum(1)).t()
        probs_lr = torch.nn.Softmax(dim=1)(fmodel(sample[0]))
        loss_6 = kl_divergence(probs_graphical, probs_lr)
        loss = loss_1 + loss_2 + loss_4 + loss_6 + loss_3 + loss_5 + prec_loss
        # print('loss --> ', loss.item())
        diffopt.step(loss)
        # print('x_valid.shape',x_valid.shape)
        # print('y_valid.shape',y_valid.shape)
        gm_val_loss = log_likelihood_loss_supervised(fmodel.theta, fmodel.pi_y, fmodel.pi, \
            y_valid,l_valid, s_valid, k,n_classes,continuous_mask, fmodel.wts)
        sup_val_loss = supervised_criterion(fmodel(x_valid), y_valid)
        valid_loss = lam1 * sup_val_loss + (1-lam1)*gm_val_loss
         # + 1e-20 * torch.norm(list(fmodel.parameters(time=0))[0], p=1)
        grad_all = torch.autograd.grad(valid_loss, list(fmodel.parameters(time=0))[0], \
            only_inputs=True)[0]
        if torch.norm(grad_all, p=2) != 0:
            temp_wts = torch.clamp(wts-5*(grad_all/torch.norm(grad_all, p=2)), min=0, max=1)
        else:
            temp_wts = wts
        return temp_wts

def rewt_lfs1(sample, lr_model, theta, pi_y, pi, wts):
    wts_param = torch.nn.Parameter(wts, requires_grad=True)
    lr_model.register_parameter("wts", wts_param)
    theta_param = torch.nn.Parameter(theta, requires_grad=True)
    lr_model.register_parameter("theta", theta_param)
    pi_y_param = torch.nn.Parameter(pi_y, requires_grad=True)
    lr_model.register_parameter("pi_y", pi_y_param)
    pi_param = torch.nn.Parameter(pi, requires_grad=True)
    lr_model.register_parameter("pi", pi_param)
    # print(lr_model.linear.weight)
    if feat_model == 'lr':
        optimizer = torch.optim.Adam([
                {'params': lr_model.linear.weight},# linear_1.parameters()},
                 # {'params': lr_model['params']['linear.bias']},
                # {'params':lr_model.linear_2.parameters()},
                # {'params':lr_model.out.parameters()},
                {'params': [lr_model.theta, lr_model.pi, lr_model.pi_y], 'lr': 0.01, 'weight_decay':0}
            ], lr=1e-4)
    elif feat_model =='nn':
        optimizer = torch.optim.Adam([
                {'params': lr_model.linear_1.parameters()},
                {'params':lr_model.linear_2.parameters()},
                {'params':lr_model.out.parameters()},
                {'params': [lr_model.theta, lr_model.pi, lr_model.pi_y], 'lr': 0.01, 'weight_decay':0}
            ], lr=1e-4)
    with higher.innerloop_ctx(lr_model, optimizer) as (fmodel, diffopt):
        supervised_criterion = torch.nn.CrossEntropyLoss()
        optimizer.zero_grad()
        supervised_indices = sample[4].nonzero().view(-1)
        unsupervised_indices = (1 - sample[4]).nonzero().squeeze()
        if (sys.argv[2] == 'l1'):
            if len(supervised_indices) > 0:
                loss_1 = supervised_criterion(fmodel(sample[0][supervised_indices]), sample[1][supervised_indices])
            else:
                loss_1 = 0
        else:
            loss_1 = 0
        unsupervised_lr_probability = torch.nn.Softmax(dim=1)(fmodel(sample[0][unsupervised_indices]).view(-1, n_classes))
        if sys.argv[3] == 'l2':
            loss_2 = entropy(unsupervised_lr_probability)
        else:
            loss_2 = 0
        y_pred_unsupervised = np.argmax(
            probability(fmodel.theta, fmodel.pi_y, fmodel.pi, sample[2][unsupervised_indices], sample[3][unsupervised_indices], k, n_classes,
                        continuous_mask, fmodel.wts, device=device).cpu().detach().numpy(), 1)
        if sys.argv[4] == 'l3':
            loss_3 = supervised_criterion(fmodel(sample[0][unsupervised_indices]), torch.tensor(y_pred_unsupervised))
        else:
            loss_3 = 0

        if sys.argv[5] == 'l4':
            if len(supervised_indices) > 0:
                loss_4 = log_likelihood_loss_supervised(fmodel.theta, fmodel.pi_y, fmodel.pi, sample[1][supervised_indices],
                                                        sample[2][supervised_indices], sample[3][supervised_indices], k,
                                                        n_classes,
                                                        continuous_mask, fmodel.wts, device)
            else:
                loss_4 = 0
        else:
            loss_4 = 0

        if sys.argv[6] == 'l5':
            loss_5 = log_likelihood_loss(fmodel.theta, fmodel.pi_y, fmodel.pi, sample[2][unsupervised_indices], sample[3][unsupervised_indices],
                                     k, n_classes, continuous_mask, fmodel.wts, device)
        else:
            loss_5 = 0

        if sys.argv[8] == 'qg':
            prec_loss = precision_loss(fmodel.theta, k, n_classes, a, fmodel.wts, device=device)
        else:
            prec_loss = 0
        probs_graphical = probability(fmodel.theta, fmodel.pi_y, fmodel.pi, sample[2], sample[3], k, n_classes, continuous_mask, fmodel.wts, device)
        probs_graphical = (probs_graphical.t() / probs_graphical.sum(1)).t()
        probs_lr = torch.nn.Softmax(dim=1)(fmodel(sample[0]))
        if sys.argv[7] == 'l6':
            loss_6 = kl_divergence(probs_graphical, probs_lr)
        else:
            loss_6 = 0
        loss = loss_1 + loss_2 + loss_4 + loss_6 + loss_3 + loss_5 + prec_loss
        # print('loss --> ', loss.item())
        diffopt.step(loss)
        # print('x_valid.shape',x_valid.shape)
        # print('y_valid.shape',y_valid.shape)
        gm_val_loss = log_likelihood_loss_supervised(fmodel.theta, fmodel.pi_y, fmodel.pi, \
            y_valid.to(device),l_valid.to(device), s_valid.to(device), k, n_classes,continuous_mask, fmodel.wts, device)
        sup_val_loss = supervised_criterion(fmodel(x_valid.to(device)), y_valid.to(device))
        valid_loss = lam1 * sup_val_loss + (1-lam1)*gm_val_loss
         # + 1e-20 * torch.norm(list(fmodel.parameters(time=0))[0], p=1)
        grad_all = torch.autograd.grad(valid_loss, list(fmodel.parameters(time=0))[0], \
            only_inputs=True)[0]
        if torch.norm(grad_all, p=2) != 0:
            temp_wts = torch.clamp(wts-5*(grad_all/torch.norm(grad_all, p=2)), min=0, max=1)
        else:
            temp_wts = wts
        return temp_wts


if mode != '':
    fname = dset_directory + "/" + mode + "_d_processed.p"
    print('fname is ', fname)
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
y_valid = torch.tensor(objs[3]).long()
l_valid = torch.tensor(objs[2]).long()
s_valid = torch.tensor(objs[2]).double()

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
y_test = torch.tensor(objs1[3]).long()
l_test = torch.tensor(objs1[2]).long()
s_test = torch.tensor(objs1[2]).double()


n_features = x_supervised.shape[1]

# Labeling Function Classes
# k = torch.from_numpy(np.array([0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0])).long()
#lf_classes_file = sys.argv[11]


fname = dset_directory + '/' + mode + '_k.npy'
k = torch.from_numpy(np.load(fname)).to(device=device).long()
##Needs to be removed##
n_lfs = len(k)
print('LFs are ',k)
print('no of lfs are ', n_lfs)

# a = torch.ones(n_lfs).double() * 0.9
# print('before ',a)
if qg_available:
    a = torch.from_numpy(np.load(dset_directory+'/prec.npy')).to(device=device).double()
else:
    # a = torch.ones(n_lfs).double() * 0.9

    prec_lfs=[]
    for i in range(n_lfs):
       correct = 0
       for j in range(len(y_valid)):
           if y_valid[j] == l_valid[j][i]:
               correct+=1
       prec_lfs.append(correct/len(y_valid))
    a = torch.tensor(prec_lfs, device=device).double()

# n_lfs = int(len(k))
# print('number of lfs ', n_lfs)
# a = torch.ones(n_lfs).double() * 0.9
continuous_mask = torch.zeros(n_lfs, device=device).double()


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
conf.n_units = num_runs
final_score_gm, final_score_lr, final_score_gm_val, final_score_lr_val = [],[],[],[]
final_score_lr_prec, final_score_lr_recall, final_score_gm_prec, final_score_gm_recall = [],[],[],[]
for lo in range(0,num_runs):
    pi = torch.ones((n_classes, n_lfs), device=device).double()
    pi.requires_grad = True

    theta = torch.ones((n_classes, n_lfs), device=device).double() * 1
    theta.requires_grad = True

    pi_y = torch.ones(n_classes, device=device).double()
    pi_y.requires_grad = True

    if feat_model == 'lr':
        lr_model = LogisticReg(n_features, n_classes).to(device=device)
    elif feat_model =='nn':
        n_hidden = 512
        lr_model = DeepNet(n_features, n_hidden, n_classes).to(device=device)
    else:
        print('Please provide feature based model : lr or nn')
        exit()

    wandb.watch(lr_model)
    optimizer = torch.optim.Adam([{"params": lr_model.parameters()}, {"params": [pi, pi_y, theta]}], lr=0.001)
    optimizer_lr = torch.optim.Adam(lr_model.parameters(), lr=lr_fnetwork)
    optimizer_gm = torch.optim.Adam([theta, pi, pi_y], lr=lr_gm, weight_decay=0)
    # optimizer = torch.optim.Adam([theta, pi, pi_y], lr=0.01, weight_decay=0)
    supervised_criterion = torch.nn.CrossEntropyLoss()


    dataset = TensorDataset(x_train, y_train, l, s, supervised_mask)

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    save_folder = sys.argv[1]
    print('num runs are ', sys.argv[1], num_runs)
    best_score_lr,best_score_gm,best_epoch_lr,best_epoch_gm,best_score_lr_val, best_score_gm_val = 0,0,0,0,0,0
    best_score_lr_prec,best_score_lr_recall ,best_score_gm_prec,best_score_gm_recall= 0,0,0,0

    stop_pahle, stop_pahle_gm = [], []
    
    # weights = torch.ones(k.shape[0])*(1/k.shape[0])
    weights = torch.ones(k.shape[0], device=device)*0.5

    for epoch in range(100):
        lr_model.train()
        #print(epoch)
        for batch_ndx, sample in enumerate(loader):
            for i in range(len(sample)):
                sample[i] = sample[i].to(device=device)
            if feat_model == 'lr':
                lr_model1 = LogisticReg(n_features, n_classes).to(device=device)
            elif feat_model =='nn':
                n_hidden = 512
                lr_model1 = DeepNet(n_features, n_hidden, n_classes).to(device=device)
            # lr_model1 = DeepNet(n_features, n_hidden, n_classes)
            lr_model1.load_state_dict(copy.deepcopy(lr_model.state_dict()))
            theta1 = copy.deepcopy(theta)
            pi_y1 = copy.deepcopy(pi_y)
            pi1 = copy.deepcopy(pi)
            weights = rewt_lfs1(sample, lr_model1, theta1, pi_y1, pi1, weights)
            optimizer_lr.zero_grad()
            optimizer_gm.zero_grad()

            unsup = []
            sup = []
            supervised_indices = sample[4].nonzero().view(-1)
            # unsupervised_indices = indices  ## Uncomment for entropy
            unsupervised_indices = (1-sample[4]).nonzero().squeeze()


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
                y_pred_unsupervised = np.argmax(probability(theta, pi_y, pi, sample[2][unsupervised_indices],\
                 sample[3][unsupervised_indices], k, n_classes,continuous_mask, weights, device=device).detach().numpy(), 1)
                loss_3 = supervised_criterion(lr_model(sample[0][unsupervised_indices]),\
                 torch.tensor(y_pred_unsupervised))
            else:
                loss_3 = 0

            if (sys.argv[5] == 'l4' and len(supervised_indices) > 0):
                loss_4 = log_likelihood_loss_supervised(theta, pi_y, pi, sample[1][supervised_indices], \
                    sample[2][supervised_indices], sample[3][supervised_indices],\
                     k, n_classes, continuous_mask, weights, device)
            else:
                loss_4 = 0

            if(sys.argv[6] =='l5'):
                loss_5 = log_likelihood_loss(theta, pi_y, pi, sample[2][unsupervised_indices], \
                    sample[3][unsupervised_indices], k, n_classes, continuous_mask, weights, device)
            else:
                loss_5 =0

            if(sys.argv[7] =='l6'):
                if(len(supervised_indices) >0):
                    supervised_indices = supervised_indices.tolist()
                    probs_graphical = probability(theta, pi_y, pi, torch.cat([sample[2][unsupervised_indices], sample[2][supervised_indices]]),\
                    torch.cat([sample[3][unsupervised_indices],sample[3][supervised_indices]]), k, n_classes, continuous_mask, weights, device=device)
                else:
                    probs_graphical = probability(theta, pi_y, pi,sample[2][unsupervised_indices],sample[3][unsupervised_indices],\
                         k, n_classes, continuous_mask, weights, device=device)
                probs_graphical = (probs_graphical.t() / probs_graphical.sum(1)).t()
                probs_lr = torch.nn.Softmax()(lr_model(sample[0]))
                loss_6 = kl_divergence(probs_lr, probs_graphical)
                # loss_6 = kl_divergence(probs_graphical, probs_lr) #original version

            else:
                loss_6= 0
            # loss_6 = - torch.log(1 - probs_graphical * (1 - probs_lr)).sum(1).mean()
            if(sys.argv[8] =='qg'):
                prec_loss = precision_loss(theta, k, n_classes, a, weights, device=device)
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
        if epoch %5 ==0:
            y_pred = np.argmax(probability(theta, pi_y, pi, l_test.to(device), s_test.to(device), k, n_classes, continuous_mask, weights, device=device).cpu().detach().numpy(), 1)

            if metric=='accuracy':
                lr_prec,lr_recall,gm_prec,gm_recall = 0,0,0,0
                gm_acc = score(y_test, y_pred)
            else:
                gm_acc = score(y_test, y_pred, average=metric_avg)
                gm_prec = prec_score(y_test, y_pred, average=metric_avg)
                gm_recall = recall_score(y_test, y_pred, average=metric_avg)

        #Valid
        y_pred = np.argmax(probability(theta, pi_y, pi, l_valid.to(device), s_valid.to(device), k, n_classes, continuous_mask, weights, device=device).cpu().detach().numpy(), 1)
        # gm_valid_acc = score(y_valid, y_pred, average="macro")
        if metric=='accuracy':
            lr_prec,lr_recall,gm_prec,gm_recall = 0,0,0,0
            gm_valid_acc = score(y_valid, y_pred)
        else:
            gm_valid_acc = score(y_valid, y_pred, average="macro")

        #LR Test
        if epoch %5 ==0:
            probs = torch.nn.Softmax()(lr_model(x_test.to(device)))
            y_pred = np.argmax(probs.cpu().detach().numpy(), 1)
            # lr_acc =score(y_test, y_pred, average="macro")
            # if name_dset =='youtube' or name_dset=='census' or name_dset =='sms':
            if metric=='accuracy':
                lr_prec,lr_recall,gm_prec,gm_recall = 0,0,0,0
                lr_acc =score(y_test, y_pred)

            else:
                lr_acc =score(y_test, y_pred, average=metric_avg)
                lr_prec = prec_score(y_test, y_pred, average=metric_avg)
                lr_recall = recall_score(y_test, y_pred, average=metric_avg)
        #LR Valid
        probs = torch.nn.Softmax()(lr_model(x_valid.to(device)))
        y_pred = np.argmax(probs.cpu().detach().numpy(), 1)
        # if name_dset =='youtube' or name_dset=='census' or name_dset =='sms':
        if metric=='accuracy':
            lr_valid_acc =score(y_valid, y_pred)
            lr_prec,lr_recall,gm_prec,gm_recall = 0,0,0,0
        else:
            lr_valid_acc =score(y_valid, y_pred, average=metric_avg)

        # lr_valid_acc = score(y_valid, y_pred, average="macro")
        if epoch %5 ==0:
            print("Epoch: {}\t Test GM accuracy_score: {}".format(epoch, gm_acc ))
    #        print("Epoch: {}\tGM accuracy_score(Valid): {}".format(epoch, gm_valid_acc))
            print("Epoch: {}\tTest LR accuracy_score: {}".format(epoch, lr_acc ))
 #       print("Epoch: {}\tLR accuracy_score(Valid): {}".format(epoch, lr_valid_acc))
        wname = "Run_"+str(lo)+" LR valid score"
        wnamegm = 'Run_' + str(lo) + ' GM valid score'
        wandb.log({wname:lr_valid_acc,
            wnamegm:gm_valid_acc,'custom_step':epoch})
        if epoch > 5 and gm_valid_acc >= best_score_gm_val and gm_valid_acc >= best_score_lr_val:
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


        if epoch > 5 and lr_valid_acc >= best_score_lr_val and lr_valid_acc >= best_score_gm_val:
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



        # if len(stop_pahle) > 10 and len(stop_pahle_gm) > 10 and (all(best_score_lr_val >= k for k in stop_pahle) or \
        # all(best_score_gm_val >= k for k in stop_pahle_gm)):
        if  len(stop_pahle) > 10 and len(stop_pahle_gm) > 10 and (all(best_score_lr_val >= k for k in stop_pahle)):
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
    # print('Best Precision LR', best_score_lr_prec)
    # print('Best Recall LR', best_score_lr_recall)
    print('Best Epoch GM', best_epoch_gm)
    # print('Best Precision GM ', best_score_gm_prec)
    # print('Best Recall GM ', best_score_gm_recall)
    print("Run \t",lo, "Epoch, GM, LR \t", best_score_gm, best_score_lr)
    print("Run \t",lo, "GM Val, LR Val \t", best_score_gm_val, best_score_lr_val)
    final_score_gm.append(best_score_gm)
    final_score_lr.append(best_score_lr)
    final_score_lr_prec.append(best_score_lr_prec)
    final_score_lr_recall.append(best_score_lr_recall)

    final_score_gm_prec.append(best_score_gm_prec)
    final_score_gm_recall.append(best_score_gm_recall)

    final_score_gm_val.append(best_score_gm_val)
    final_score_lr_val.append(best_score_lr_val)


wandb.log({'test_lr':np.mean(final_score_lr),'test_gm':np.mean(final_score_gm)})#wandb

print("===================================================")
print("TEST Averaged scores LR", np.mean(final_score_lr))
print("TEST Precision averaged scores LR", np.mean(final_score_lr_prec))
print("TEST Recall averaged scores LR", np.mean(final_score_lr_recall))
print("===================================================")
print("TEST Averaged scores GM",  np.mean(final_score_gm))
print("TEST Precision averaged scores GM", np.mean(final_score_gm_prec))
print("TEST Recall averaged scores GM", np.mean(final_score_gm_recall))
print("===================================================")
print("VALIDATION Averaged scores are GM,LR", np.mean(final_score_gm_val), np.mean(final_score_lr_val))
print("TEST STD GM,LR", np.std(final_score_gm), np.std(final_score_lr))
print("VALIDATION STD GM,LR", np.std(final_score_gm_val), np.std(final_score_lr_val))
wt = np.asarray(weights)
np.save(os.path.join(dset_directory, 'weights'), wt)
print('Sorted weights ', wt.argsort())

wandb.log({'test_mean_LR ':np.mean(final_score_lr), 'test_mean_GM': np.mean(final_score_gm)}) #wandb
wandb.log({'test_STD_LR ':np.std(final_score_lr), 'test_STD_GM': np.std(final_score_gm)}) #wandb



