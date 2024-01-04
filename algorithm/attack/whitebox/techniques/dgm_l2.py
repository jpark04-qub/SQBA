import torch
import numpy as np
import torch.optim as optim
import torch.nn as nn
from ...base import Base
from ...utility import Utility as util


class DGM_L2(Base):
    def __init__(self, device, model, eps=0.01, alp=0.005, iter=100, c=0.3, stop=False, min=0, max=1):
        super().__init__()
        self.device = device
        self.model = model
        self.eps = eps
        self.lr = alp
        self.iter = iter
        self.stop = stop
        self.c = c
        self.min = min
        self.max = max
        self.label = torch.tensor([0])
        self.targeted = False

    def f(self, x):
        x.requires_grad = True
        p = self.model(x)
        _, l = torch.sort(p, descending=True)
        if self.targeted:
            target = self.label
        else:
            target = l[0, 1]

        return l[0, 0], target, p

    def tune_adv(self, image, label, adv_init):
        if self.lr == 0:
            return adv_init, 0

        if self.targeted:
            l_ref = self.label
        else:
            l_ref = label

        optimizer = optim.Adam([adv_init], lr=self.lr)
        i = 0
        while 1:
            adv = adv_init.detach().clone()
            # dist = torch.dist(adv, image, 2)
            adv_init.requires_grad = True
            cost = nn.MSELoss(reduction='sum')(adv_init, image)

            optimizer.zero_grad()
            cost.backward()
            optimizer.step()

            l_i, _, _ = self.f(torch.clamp(adv_init, self.min, self.max).detach())
            if self.targeted:
                if l_ref != l_i:
                    break
            else:
                if l_ref == l_i:
                    break
            i = i + 1

        return adv, i

    def core(self, image, label):
        x_i = image.detach().clone()
        x_i.requires_grad = True
        l_true, l_target, p = self.f(x_i)
        if self.targeted and l_true == label:
            adv = image.detach().clone()
            perturb = torch.zeros_like(image, requires_grad=False)
            return adv, perturb, 0, 0
        l_i = l_true
        converged = False
        iter0 = 0
        i = 0

        disp = util(10)

        while i < self.iter:
            disp.get(p)
            x_j = x_i.detach().clone()
            for j in range(2):
                if j == 0:
                    temp = (p[0, l_target] / (p[0, l_true] + p[0, l_target]))
                    c = 1 / torch.exp(4 * temp)
                    if c > self.c:
                        c = self.c
                    w_j = -torch.autograd.grad(p[0, l_true], x_i, retain_graph=True, create_graph=False)[0]
                else:
                    c = 1 - c

                    if c < 0 or c > 1:
                        print("Error wrong c vakue \n")

                    w_j = torch.autograd.grad(p[0, l_target], x_i, retain_graph=True, create_graph=False)[0]

                x_j = x_j + self.eps * c * (w_j / torch.max(torch.abs(w_j)))
                x_j = torch.clamp(x_j, self.min, self.max).detach()

            x_i = x_j.detach().clone()
            i += 1
            l_i, l_target, p = self.f(x_i)
            if self.targeted:
                if l_i == l_target:
                    if not converged:
                        converged = True
                        i = max(self.iter-round(i/2.25), 0)
            else:
                if l_i != l_true:
                    l_target = l_i
                    if not converged:
                        converged = True
                        i = max(self.iter-round(i/2.25), 0)

            if self.stop and l_i != l_true:
                break

            iter0 += 1

        if self.stop and i == self.steps - 1:
            print("NewAlg_l2 attack is not converged")

        disp.show()

        adv = x_i
        adv, iter1 = self.tune_adv(image, l_true, adv)

        adv.requires_grad = False
        adv = torch.clamp(adv, self.min, self.max)

        return adv, iter0, iter1

    def untarget(self, image, label):
        self.targeted = False

        adv = torch.zeros_like(image, requires_grad=False)
        perturb = torch.zeros_like(image, requires_grad=False)

        iter0 = 0
        iter1 = 0
        for b in range(image.shape[0]):
            img = image[b:b + 1, :, :, :].detach().clone()
            adv[b:b+1, :, :, :], iter0, iter1 = self.core(img, label[b])
            perturb[b:b+1, :, :, :] = adv[b:b+1, :, :, :]-img

        return adv, iter0, iter0, iter1

    def target(self, image, label):
        self.targeted = True
        self.label = label

        adv = torch.zeros_like(image, requires_grad=False)
        perturb = torch.zeros_like(image, requires_grad=False)

        iter0 = 0
        iter1 = 0
        for b in range(image.shape[0]):
            img = image[b:b + 1, :, :, :].detach().clone()
            adv[b:b+1, :, :, :], iter0, iter1 = self.core(img, label[b])
            perturb[b:b+1, :, :, :] = adv[b:b+1, :, :, :]-img

        return adv, iter0, iter0, iter1

