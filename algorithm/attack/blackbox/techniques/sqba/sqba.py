import torch
import numpy as np
from ....base import HardLabelAttackBase

"""
Jeonghwan Park, Niall McLaughlin, Paul Miller
Hard-label based small query black-box adversarial attack
IEEE/CVF Winter Conference on Applications of Computer Vision, 2024
"""
class SQBA(HardLabelAttackBase):
    def __init__(self, device, model, sub_model=None, lossF=None, lp='l2', q_budgets=1000, stop=True):
        super().__init__(device=device,
                         targeted=False,
                         model=model,
                         lp=lp,
                         q_budgets=q_budgets,
                         stop=stop)
        self.name = 'sqba'

        self.sub_model = sub_model
        self.lossF = lossF

        self.min = 0
        self.max = 1

        self.init_eval = 100
        self.theta = 0.001
        self.threshold = 0.001
        self.iter_cnt = 0

        self.min_randoms = 10

        self.d_prev = 0

        self.white_only = True

        self.x = 0

    def surrogate_model(self, x):
        self.query_cnt2 += 1
        return self.sub_model(x)

    def check_adversary(self, x):
        pred = self.prediction(x.reshape(self.shape))
        flag = False
        if self.targeted:
            if pred == self.label:
                flag = True
        else:
            if pred != self.label:
                flag = True
        return flag

    def _interpolate(self, current_sample, alpha):

        if self.lp == 'l2':
            result = (1 - alpha) * self.x + alpha * current_sample
        else:
            lb = torch.min((self.x - alpha).flatten())
            hb = torch.max((self.x + alpha).flatten())
            result = torch.clamp(current_sample, lb, hb)

        return result

    def _compute_delta(self, current_sample):
        if self.iter_cnt == 0:
            return 0.1 * (self.max - self.min)

        if self.lp == 'l2':
            dist = torch.norm(self.x - current_sample)
            delta = np.sqrt(np.prod(current_sample.size())) * self.theta * dist
        else:
            dist = torch.max(torch.abs(self.x - current_sample))
            delta = np.prod(current_sample.size()) * self.theta * dist
            delta = delta + 1e-17

        return delta

    def _binary_search(self, current_sample, threshold):
        # First set upper and lower bounds as well as the threshold for the binary search
        if self.lp == 'l2':
            upper_bound = torch.tensor(1)
            lower_bound = torch.tensor(0)

            if threshold is None:
                threshold = self.theta

        else:
            upper_bound = torch.max(torch.abs(self.x - current_sample))
            upper_bound = upper_bound.cpu()
            lower_bound = torch.tensor(0)

            if threshold is None:
                threshold = np.minimum(upper_bound * self.theta, self.theta)

        # Then start the binary search
        while (upper_bound - lower_bound) > threshold:
            # Interpolation point
            alpha = (upper_bound + lower_bound) / 2.0
            interpolated_sample = self._interpolate(current_sample, alpha)

            # Update upper_bound and lower_bound
            satisfied = self.check_adversary(interpolated_sample)

            lower_bound = torch.where(torch.tensor(satisfied == 0), alpha, lower_bound)
            upper_bound = torch.where(torch.tensor(satisfied == 1), alpha, upper_bound)

            if self.query_cnt > self.query_max:
                break

        result = self._interpolate(current_sample, upper_bound)

        return result

    def dgm(self, x, yn, yp):
        x.requires_grad = True

        output = self.surrogate_model(x)

        loss = self.lossF(output, yn)
        gn = torch.autograd.grad(loss, x, retain_graph=True, create_graph=False)[0]
        loss = self.lossF(output, yp)
        gp = torch.autograd.grad(loss, x, retain_graph=False, create_graph=False)[0]

        x.requires_grad = False

        eps = 0.8
        gn = gn.sign() * (1)
        gp = gp.sign() * (-1)

        perturb = (eps * gn) + ((1 - eps) * gp)

        return perturb

    def dgm_wrap(self, x):
        output = self.surrogate_model(x)

        # choose positive direction with the highest probability class
        _, i_p = torch.sort(output, descending=True)
        for i in range(i_p.size()[1]):
            if i_p[0, i] != self.label.reshape([1]):
                yp = i_p[0, i].reshape(1)
                break
        g = self.dgm(x, self.label.reshape([1]), yp)

        return g

    def make_sample(self, x_adv):
        x_adv = x_adv.reshape(self.shape)
        g = self.dgm_wrap(x_adv)
        g = g.flatten()

        return g

    def white_grad(self, x_adv, delta):
        v = x_adv - self.x

        delta = delta*2
        num = 10
        bias = 0.1
        a = (1 - (bias * 2)) / (num - 1)
        r = np.zeros(num)
        for i in range(num):
            r[i] = bias + (a * i)

        d_list = torch.zeros([1, num])
        a_list = torch.zeros([1, num])
        g_list = []
        x_list = []
        for i in range(num):
            g_t = self.make_sample(self.x + v * r[i])

            aa = g_t.flatten()
            bb = v.flatten()
            cc = torch.matmul(aa, bb)
            angle = cc / (torch.norm(aa) * torch.norm(bb))

            x_t = torch.clamp((x_adv + (delta * g_t)), self.min, self.max)
            d_t = torch.norm(self.x - x_t)

            x_list.append(x_t)
            g_list.append(g_t)
            d_list[0, i] = d_t
            a_list[0, i] = angle

        d_sort, indices = torch.sort(d_list)
        for i in range(num):
            idx = indices[0, i]
            if self.check_adversary(x_list[idx]):
                break
        else:
            idx = indices[0, 0]
            print("failed adversarial")

        return g_list[idx]

    def _init_sample(self):
        x = self.x.clone()
        x_adv = x.clone()
        initial_sample = None

        if self.targeted:
            print("targeted attack is not supported yet")
            return None
        else:
            for _ in range(self.init_eval):
                x_adv = self.make_sample(x_adv)

                label = self.prediction(x_adv.reshape(self.shape))
                if label != self.label:
                    # Binary search to reduce the l2 distance to the original image
                    x_adv = self._binary_search(x_adv, self.threshold * 10)
                    initial_sample = x_adv
                    break
            else:
                print('failed - init sample')

        return initial_sample

    def white_update(self, x_adv, delta):
        u_t = self.white_grad(x_adv, delta)

        x_t = torch.clamp((x_adv + delta * u_t), self.min, self.max)
        dh = (x_t - x_adv) / delta
        g = dh / torch.norm(dh)

        return g

    def black_update(self, x_adv, delta, num_eval):
        x_adv = x_adv.clone().flatten()

        if delta == 0 or num_eval == 0:
            print("delta {:.3f}, num_eval {}".format(delta, num_eval))

        rnd_noise_shape = [num_eval] + list(x_adv.size())
        if self.lp == 'l2':
            rnd_noise = torch.randn(rnd_noise_shape).to(self.device)
        else:
            rnd_noise = torch.rand(rnd_noise_shape)

        # Normalize random noise to fit into the range of input data
        rnd_noise = rnd_noise / torch.sqrt(
            torch.sum(rnd_noise ** 2, axis=tuple(range(len(rnd_noise_shape)))[1:], keepdims=True))

        eval_samples = torch.clamp(x_adv + delta * rnd_noise, self.min, self.max)
        rnd_noise = (eval_samples - x_adv) / delta

        satisfied = torch.zeros(num_eval)
        for i in range(num_eval):
            satisfied[i] = self.check_adversary(eval_samples[i, :].reshape(self.shape))

        f_val = 2 * satisfied - 1.0
        f_val = f_val.to(self.device)

        if torch.mean(f_val) == 1.0:
            grad = torch.mean(rnd_noise, axis=0)
        elif torch.mean(f_val) == -1.0:
            grad = -torch.mean(rnd_noise, axis=0)
        else:
            m = torch.mean(f_val)
            f_val -= m
            f_val = f_val.reshape([len(f_val), 1])
            grad = torch.mean(f_val * rnd_noise, axis=0)

        # Compute update
        if self.lp == 'l2':
            g = grad / torch.norm(grad)
        else:
            g = np.sign(grad)

        if torch.isnan(g).any():
            print("result Nan - {:.5f}".format(torch.norm(grad)))

        return g

    def compute_update(self, x_adv, delta):
        if self.white_only:
            g = self.white_update(x_adv, delta)
        else:
            num_eval = int(self.min_randoms * np.sqrt(self.iter_cnt + 1))
            num_eval = min(int(num_eval), np.abs(self.query_max - self.query_cnt))
            g = self.black_update(x_adv, delta, num_eval)

        return g

    def perturb(self, x_best):
        # intermediate adversarial example
        x_prev = x_best.clone()
        x_adv = x_best.clone()

        success_cnt = 0

        delta = self._compute_delta(x_adv)

        # Then run binary search
        x_adv = self._binary_search(x_adv, self.threshold)
        # Next compute the number of evaluations and compute the update

        update = self.compute_update(x_adv, delta)

        # Finally run step size search by first computing epsilon
        if self.lp == 'l2':
            dist = torch.norm(self.x - x_adv)
        else:
            dist = np.max(abs(self.x - x_adv))

        epsilon = 2.0 * dist / np.sqrt(self.iter_cnt + 1)

        while True:
            epsilon /= 2.0
            x_c = x_adv + epsilon * update
            success = self.check_adversary(torch.clamp(x_c, self.min, self.max))
            if success:
                break

        if self.white_only:
            if dist < self.d_prev:
                self.d_prev = dist
                x_best = x_adv
        else:
            x_best = x_adv

        if epsilon < 1.0:
            # stop using updates from surrogate model
            self.white_only = False
            x_best = x_adv

        # Update current sample
        if success is True:
            x_best = torch.clamp(x_c, self.min, self.max)
            success_cnt += 1

        #print("{} - {:.4f} {:.4f} {:.4f}".format(self.iter_cnt, dist, epsilon, delta))
        # Update current iteration
        self.iter_cnt += 1

        if success_cnt == 0:
            print('failed - converge')

        if torch.isnan(x_best).any():  # pragma: no cover
            x_best = x_prev

        return x_best

    def attack(self, sm, x_best):
        if sm == 0:
            x_best = self._init_sample()
            if x_best is not None:
                sm = 1
            self.query_cnt2 = self.query_cnt
        elif sm == 1:
            x_best = self.perturb(x_best)

        return sm, x_best

    def setup(self, x):
        self.shape = x.size()
        x_ = x.detach().clone().flatten()
        self.x = x_
        x_best = torch.zeros_like(self.x)

        self.d_prev = torch.inf

        # Set binary search threshold
        if self.lp == 'l2':
            self.theta = 0.01 / np.sqrt(np.prod(self.shape))
        else:
            self.theta = 0.01 / np.prod(self.shape)

        return x_best

    def core(self, image):
        x_best = self.setup(image)

        # Generate the adversarial samples
        x_adv, queries0, queries1 = self.run(x_best)

        return x_adv, queries0, queries1

    def untarget(self, image, label):
        self.targeted = False
        self.label = label
        self.min = torch.min(image.flatten())
        self.max = torch.max(image.flatten())

        img = image.detach().clone()
        adv, q0, q1 = self.core(img)

        return adv[0], q0[0], q1[0], 0



