import torch
import numpy as np
import torchvision
import algorithm.attack as attack
from utilities.loss_function import list as lossf_list

import matplotlib.pyplot as plt
import time


def make_grid(sample):
    img = torchvision.utils.make_grid(sample)
    return img.detach().numpy()


class Configuration:
    def __init__(self, on=False, name="none", eps=0, eta=0, alp=0, c=0, lr=0, iter=0, sigma=0, stop=False):
        self.on = on
        self.name = name
        self.eps = eps
        self.eta = eta
        self.c = c
        self.alp = alp
        self.lr = lr
        self.iter = iter  # steps
        self.sigma = sigma
        self.stop = stop
        return


def prediction(model, x):
    output = model(x)
    _, hx = output.data.max(1)
    return hx


def test(device, classes, target_model, sub_model, test_loader):
    sub_model.to(device)

    dgm_cfg  = Configuration(False, "dgm", eps=0.005, alp=0.001, iter=300, c=0.3)
    sqba_cfg = Configuration(True, "sqba", iter=250)

    count = 0
    success_cnt = np.zeros(2).astype(int)
    queries = np.zeros(2).astype(int)
    throughput = np.zeros(2)
    predict = np.zeros(2).astype(int)
    distance = np.zeros(2)
    algorithm = []

    for data, true_class in test_loader:

        x = data.clone()
        y = true_class.clone()

        pix_min = torch.min(x.flatten()).to(device)
        pix_max = torch.max(x.flatten()).to(device)

        x = x.to(device)
        y = y.to(device)

        h = prediction(target_model, x)
        if h != y:
            continue

        idx = 0

        count += 1
        if count > 1001:
            break

        def evaluation(idx, alg, name):
            algorithm.append(name)
            t0 = time.perf_counter()
            ladv, lquery, iter0, iter1 = alg.untarget(x, y)
            t1 = time.perf_counter()

            loutput = target_model(ladv)
            lpred = loutput.max(1, keepdim=True)[1]
            predict[idx] = (lpred.item())
            queries[idx] = lquery
            throughput[idx] = (t1 - t0)
            distance[idx] = (torch.norm(torch.abs(ladv - x)) / torch.norm(x))

            if lpred.item() != y:
                success_cnt[idx] += 1

        if dgm_cfg.on:
            cfg = dgm_cfg
            alg = attack.DGM_L2(device, target_model, eps=cfg.eps, min=pix_min, max=pix_max)
            evaluation(idx, alg, cfg.name)
            idx += 1

        if sqba_cfg.on:
            cfg = sqba_cfg
            lossF = lossf_list(sub_model.loss).to(device)
            alg = attack.SQBA(device, model=target_model, sub_model=sub_model, lossF=lossF,  q_budgets=[cfg.iter], stop=False)
            evaluation(idx, alg, cfg.name)
            idx += 1

        if count == 1 or show_model == 1:
            show_model = 0
            for i in range(idx):
                print("{}[{} - {} {}] ".format(i, algorithm[i], target_model.name, sub_model.name), end='')
            print("")

        print("{}[{}]- ".format(count, classes[true_class.item()]), end='')
        for i in range(idx):
            print("{}[{}, {:.3f}, {:.3f}, {}] ".format(success_cnt[i],
                                                       classes[predict[i]], distance[i], throughput[i], queries[i]), end='')
        print("")

