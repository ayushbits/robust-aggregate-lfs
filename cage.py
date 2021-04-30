import torch
from torch.distributions.beta import Beta


def probability_y(pi_y):
    pi = torch.exp(pi_y)
    return pi / pi.sum()


def phi(theta, l):
    value = theta * torch.abs(l).double()
    #print(value)
    return value


def calculate_normalizer(theta, k, n_classes):
    z = 0
    for y in range(n_classes):
        m_y = torch.exp(phi(theta[y], torch.ones(k.shape)))
        z += (1 + m_y).prod()
    return z


def probability_l_y(theta, l, k, n_classes):
    probability = torch.zeros((l.shape[0], n_classes))
    z = calculate_normalizer(theta, k, n_classes)
    for y in range(n_classes):
        # print('l.shape ', l.shape)

        yo = phi(theta[y], l)
       # print('yo.shape', yo.shape)
        # print(yo.shape[0])
        # try:
        #yo = yo.view(-1, l.shape[0])
        
        yoo = torch.exp(yo.sum(1))

        # except:
            # print('inside except cage #32')
        # yoo = torch.exp(yo.sum())
        probability[:, y] =  yoo/ z

    return probability.double()


def probability_s_given_y_l(pi, s, y, l, k, continuous_mask, ratio_agreement=0.85, model=1, theta_process=2):
    eq = torch.eq(k.view(-1, 1), y).double().t()
    r = ratio_agreement * eq.squeeze() + (1 - ratio_agreement) * (1 - eq.squeeze())
    params = torch.exp(pi)
    probability = 1
    for i in range(k.shape[0]):
        m = Beta(r[i] * params[i], params[i] * (1 - r[i]))
        probability *= (torch.exp(m.log_prob(s[:, i].double())) * l[:, i].double() + (1 - l[:, i]).double()) * continuous_mask[i] + (1 - continuous_mask[i])
    return probability


def probability(theta, pi_y, pi, l, s, k, n_classes, continuous_mask):
    p_l_y = probability_l_y(theta, l, k, n_classes)
    p_s = torch.ones(s.shape[0], n_classes).double()
    for y in range(n_classes):
        p_s[:, y] = probability_s_given_y_l(pi[y], s, y, l, k, continuous_mask)
    return p_l_y * p_s
    # print((prob.T/prob.sum(1)).T)
    # input()
    # return prob
    # return (prob.T/prob.sum(1)).T


def log_likelihood_loss(theta, pi_y, pi, l, s, k, n_classes, continuous_mask):
    eps = 1e-8
    return - torch.log(probability(theta, pi_y, pi, l, s, k, n_classes, continuous_mask).sum(1)).sum() / s.shape[0]


def log_likelihood_loss_supervised(theta, pi_y, pi, y, l, s, k, n_classes, continuous_mask):
    eps = 1e-8
    prob = probability(theta, pi_y, pi, l, s, k, n_classes, continuous_mask)
    prob = (prob.t() / prob.sum(1)).t()
    return torch.nn.NLLLoss()(torch.log(prob), y)
    # return - torch.log(probability(theta, pi_y, pi, l, s, k, n_classes, continuous_mask)[:, y] + eps).mean()# / s.shape[0]


def precision_loss(theta, k, n_classes, a):
    n_lfs = k.shape[0]
    prob = torch.ones(n_lfs, n_classes).double()
    z_per_lf = 0
    for y in range(n_classes):
        m_y = torch.exp(phi(theta[y], torch.ones(n_lfs)))
        per_lf_matrix = torch.tensordot((1 + m_y).view(-1, 1), torch.ones(m_y.shape).double().view(1, -1), 1) - torch.eye(n_lfs).double()
        prob[:, y] = per_lf_matrix.prod(0).double()
        z_per_lf += prob[:, y].double()
    prob /= z_per_lf.view(-1, 1)
    correct_prob = torch.zeros(n_lfs)
    for i in range(n_lfs):
        correct_prob[i] = prob[i, k[i]]
    loss = a * torch.log(correct_prob).double() + (1 - a) * torch.log(1 - correct_prob).double()
    return -loss.sum()
