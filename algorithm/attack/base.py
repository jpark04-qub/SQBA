import torch
import numpy as np
from torch import Tensor as t
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torchvision

class Base:
    def __init__(self, device='cpu'):
        self.device = device

    def forward(self):
        print('Attack Base forward')

class WhiteBoxAttackBase:
    def __init__(self, device='cpu'):
        self.device = device

    def forward(self):
        print('Attack Base forward')


def random_adaptive(eta, adapt):
    m0 = torch.min(eta)
    m1 = torch.max(eta)

    mean = torch.rand(1).to(eta.device)
    if mean > 1.0:
        mean = 1.0
    # positive bound of mean - ratio of distribution variance
    bound = (m1-m0)*adapt
    mean *= bound
    # sign of mean
    sign = torch.randn(1).to(eta.device)
    sign /= torch.abs(sign)

    eta = eta + sign*mean
    return eta

def random_adaptive_sigma(eta, adapt):
    m0 = torch.min(eta)
    m1 = torch.max(eta)

    #m3 = torch.var(eta)

    sigma = torch.rand(1).to(eta.device)
    if sigma > 1.0:
        sigma = 1.0
    # positive bound of mean - ratio of distribution variance
    #bound = (m1-m0)*1.0
    sigma *= 0.25
    # sign of mean
    sign = torch.randn(1).to(eta.device)
    sign /= torch.abs(sign)

    sigma = 1+sign*sigma
    eta = eta * sigma

    #m4 = torch.var(eta)
    return eta

class SoftLabelAttackBase:
    def __init__(self, device='cpu', targeted=False, model=None, lp='l2', stop=True, q_budgets=10000, max_rho=0.1, debug=False):
        self.device = device
        self.targeted = targeted
        self.model = model
        self.lp = lp
        self.query_cnt = 0
        self.query_cnt2 = 0
        self.query_max = np.max(q_budgets)
        self.q_budgets = np.sort(q_budgets)
        self.shape = torch.zeros(4) # [batch_size x channel x width x height]
        self.label = 0
        self.stop = stop
        self.max_rho = max_rho
        self.debug = debug
        self.extension = True
        self.adapt = 1

    def model_(self, x):
        self.query_cnt += 1
        return self.model(x)

    def make_grid(self, sample):
        img = torchvision.utils.make_grid(sample)
        img = img.clone().to('cpu')
        return img.detach().numpy()

    def set_adapt(self, adapt):
        self.adapt = adapt

    def random_test(self, eta):
        eta = random_adaptive(eta, self.adapt)
        return eta

    def prediction(self, x):
        output = self.model_(x)
        _, hx = output.data.max(1)
        return hx

    def probability(self, x, y):
        output = self.model_(x)
        prob = output[0, y]
        return output, prob

    def check_adversary(self, x):
        x = x.clone()
        hx = self.prediction(x.reshape(self.shape))
        flag = False
        if self.targeted:
            if hx == self.label:
                flag = True
        else:
            if hx != self.label:
                flag = True
        return flag

    def stop_criteria(self, x, x_adv):
        if x_adv is None:
            return False
        x = x.clone().detach().reshape(self.shape)
        x_adv = x_adv.clone().detach().reshape(self.shape)
        flag = False
        eta = x_adv - x
        eta_norm = torch.norm(eta)
        rho = eta_norm / (torch.norm(x)+1e-17)
        if self.stop:
            if self.check_adversary(x_adv) and (rho <= self.max_rho):
                flag = True
        return flag

    def attack(self, sm, x_best):
        raise NotImplementedError

    def run(self, x_best):

        # Generate the adversarial samples
        x_adv = []
        queries0 = []
        queries1 = []
        sm = 0
        for i in range(len(self.q_budgets)):
            self.query_max = self.q_budgets[i]
            self.extension = True
            while True:
                sm, x_best = self.attack(sm, x_best)
                if self.stop_criteria(self.x, x_best):
                    break
                if self.query_cnt >= self.query_max:
                    break
            #self.max_rho *= 0.9
            x_best = torch.clamp(x_best, self.min, self.max)
            x_adv.append(x_best.reshape(self.shape))
            queries0.append(self.query_cnt)
            queries1.append(self.query_cnt2)

        return x_adv, queries0, queries1


class HardLabelAttackBase:
    def __init__(self, device='cpu', targeted=False, model=None, lp='l2', stop=True, q_budgets=10000, max_rho=0.1,
                 adaptive=False, debug=False):
        self.device = device
        self.targeted = targeted
        self.model = model
        self.lp = lp
        self.query_cnt = 0
        self.query_cnt2 = 0
        self.query_max = np.max(q_budgets)
        self.q_budgets = np.sort(q_budgets)
        self.shape = torch.zeros(4) # [batch_size x channel x width x height]
        self.label = 0
        self.stop = stop
        self.max_rho = max_rho
        self.adaptive = adaptive
        self.debug = debug
        self.extension = True
        self.x2_seq_idx = 0
        self.x2_data = None
        self.x2_seq = None
        self.x2_n = None
        self.x2_n_idx = 0
        self.x2_factor = 4.5
        self.adapt = 1

    def model_(self, x):
        self.query_cnt += 1
        return self.model(x)

    def make_grid(self, sample):
        img = torchvision.utils.make_grid(sample)
        img = img.clone().to('cpu')
        return img.detach().numpy()

    def set_adapt(self, adapt):
        self.adapt = adapt

    def random_test(self, eta):
        eta = random_adaptive(eta, self.adapt)
        return eta

    def adapt_type0(self, x):
        # eot
        eot_iter = 3
        # find most predicted label
        p = torch.zeros(eot_iter)
        c = torch.zeros(eot_iter)
        for i in range(eot_iter):
            output = self.model_(x)
            _, hx = output.data.max(1)
            for j in range(eot_iter):
                if p[j] == hx.item() or c[j] == 0:
                    p[j] = hx.item()
                    c[j] += 1
                    break
            if self.query_cnt >= self.query_max:
                break
        _, i = torch.sort(c, descending=True)
        return p[i[0]]

    def adapt_type1(self, x):
        # [randome noise] -> [x] -> [random noise]
        shape = self.x2_data.size()

        n = self.x2_n[self.x2_n_idx]
        self.x2_n_idx += 1
        if self.x2_n_idx >= len(self.x2_n):
            self.x2_n = torch.rand(shape[0])
            factor = 1 / self.x2_factor
            self.x2_n = self.x2_n // factor
            self.x2_n_idx = 0

        for i in range(int(n)):
            x2 = torch.randn_like(x)
            output = self.model_(x2.cuda())

        output = self.model_(x)
        _, hx = output.data.max(1)
        return hx

    def adapt_type2(self, x):
        # [n benign examples] -> [x] -> [n benign examples]
        shape = self.x2_data.size()

        n = self.x2_n[self.x2_n_idx]
        self.x2_n_idx += 1
        if self.x2_n_idx >= len(self.x2_n):
            self.x2_n = torch.rand(shape[0])
            factor = 1 / self.x2_factor
            self.x2_n = self.x2_n // factor
            self.x2_n_idx = 0

        for i in range(int(n)):
            idx = self.x2_seq[self.x2_seq_idx]
            x2 = self.x2_data[idx, :, :, :]
            x2 = x2.reshape(1, shape[1], shape[2], shape[3])
            output = self.model_(x2.cuda())
            self.x2_seq_idx += 1
            if self.x2_seq_idx >= shape[0]:
                self.x2_seq_idx = 0
                self.x2_seq = torch.randperm(shape[0])

        output = self.model_(x)
        _, hx = output.data.max(1)
        return hx

    def reset_adapt(self, pool):
        shape = pool.size()
        self.x2_data = pool
        self.x2_seq = torch.randperm(shape[0])
        self.x2_seq_idx = 0
        self.x2_n = torch.rand(shape[0])
        factor = 1/self.x2_factor
        self.x2_n = self.x2_n // factor
        self.x2_n_idx = 0

    def prediction(self, x):
        if self.adaptive:
            #hx = self.adapt_type0(x)
            hx = self.adapt_type1(x)
            #hx = self.adapt_type2(x)
        else:
            output = self.model_(x)
            _, hx = output.data.max(1)
        return hx

    def check_adversary(self, x):
        x = x.clone()
        hx = self.prediction(x.reshape(self.shape))
        flag = False
        if self.targeted:
            if hx == self.label:
                flag = True
        else:
            if hx != self.label:
                flag = True
        return flag

    def stop_criteria(self, x, x_adv):
        if x_adv is None:
            return False
        x = x.clone().detach().reshape(self.shape)
        x_adv = x_adv.clone().detach().reshape(self.shape)
        flag = False
        eta = x_adv - x
        eta_norm = torch.norm(eta)
        rho = eta_norm / (torch.norm(x)+1e-17)
        if self.stop:
            if self.check_adversary(x_adv) and (rho <= self.max_rho):
                flag = True
        return flag

    def attack(self, sm, x_best):
        raise NotImplementedError

    def run(self, x_best):

        # Generate the adversarial samples
        x_adv = []
        queries0 = []
        queries1 = []
        sm = 0
        for i in range(len(self.q_budgets)):
            self.query_max = self.q_budgets[i]
            self.extension = True
            while True:
                sm, x_best = self.attack(sm, x_best)
                if self.stop_criteria(self.x, x_best):
                    break
                if self.query_cnt >= self.query_max:
                    break
            self.max_rho *= 0.9
            if x_best is None:
                x_best = self.x
            x_best = torch.clamp(x_best, self.min, self.max)
            x_adv.append(x_best.reshape(self.shape))
            queries0.append(self.query_cnt)
            queries1.append(self.query_cnt2)

        return x_adv, queries0, queries1