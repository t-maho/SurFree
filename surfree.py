import logging
import numpy as np
import torch
import copy
import random
import tqdm
import eagerpy as ep

from scipy import interpolate

from .base import MinimizationAttack, get_criterion, get_is_adversarial
from .blended_noise import LinearSearchBlendedUniformNoiseAttack
from .surfree_utils import Basis, get_module_name, norm, get_raw
from ..devutils import atleast_kd
from ..distances import l2, linf, l0, l1


class SurFree(MinimizationAttack):

    distance = l2
    def __init__(self, steps=100, BS_gamma=0.01, BS_max_iteration=10, theta_max=30, evolution="circular", last_direction_ortho=100, rho=0.98,  T=1, with_BS_alpha=True, with_BS=False, with_interp=False):
        """[summary]

        Args:
            steps (int, optional): run steps. Defaults to 1000.
            BS_gamma ([type], optional): Binary Search Early Stop. Stop when precision is below BS_gamma. Defaults to 0.01.
            BS_max_iteration ([type], optional): Max iteration for . Defaults to 10.
            theta_max (int, optional): max theta watched to evaluate the direction. Defaults to 30.
            evolution (str, optional): Move in this direction. It can be linear or circular. Defaults to "circular".
            last_direction_ortho (int, optional): Orthogonalize with the last directions. Defaults to 100.
            rho (float, optional): Malus/Bonus factor given to the theta_max. Defaults to 0.98.
            T (int, optional): How many evaluation done to evaluated a direction. Defaults to 1.
            with_BS_alpha (bool, optional): Activate Binary Search on Theta. Defaults to True.
            with_BS (bool, optional): Activate Binary Search between adversarial and x_o. Defaults to False.
            with_interp (bool, optional): Activate Interpolation. Defaults to False.
        """
        self.BS_gamma = BS_gamma
        self.BS_max_iteration = BS_max_iteration

        self._steps = steps
        self.with_BS, self.with_interp, self.with_BS_alpha = with_BS, with_interp, with_BS_alpha

        self._evolution = evolution
        self.best_advs = None
        self.theta_max = theta_max
        self.rho = rho
        self.T = T

        assert self.rho <= 1 and self.rho > 0

        self.last_direction_ortho = last_direction_ortho

        self._directions_ortho = {}
        self._first_direction = {}
        self._nqueries = 0

    def get_nqueries(self):
        return self._nqueries

    def run(self, model, inputs, criterion, early_stop=None, starting_points=None, basis_params=None, init_attack_step=10):
        self._nqueries = {i: 0 for i in range(len(inputs))}
        self._model = model
        originals, restore_type = ep.astensor_(inputs)

        self.device = originals.raw.device if originals.raw.is_cuda else None

        # Load Basis
        if basis_params is None:
            basis_params = {}
        basis_params["device"] = self.device
        self._basis = Basis(originals, criterion.labels, **basis_params)


        self.theta_max = [self.theta_max] * len(originals)
        criterion = get_criterion(criterion)

        # Get Starting Point
        if starting_points is not None:
            is_adversarial = get_is_adversarial(criterion, model)
            assert is_adversarial(starting_points).all()
            self.best_advs = starting_points
        elif starting_points is None:
            init_attack = LinearSearchBlendedUniformNoiseAttack(steps=init_attack_step)
            self.best_advs = init_attack.run(model, originals, criterion)
        else:
            raise ValueError("starting_points {} doesn't exist.".format(starting_points))
        
        sp_l2 = self.distance(self.best_advs, originals)
        print("Starting point found at distance:")
        for i, dist in enumerate(sp_l2):
            print("\t- Image {}: {}".format(i, dist))

        # Define The first direction : x_o with the starting point
        self._first_direction = self.best_advs - originals
        self._first_direction = self._first_direction.flatten(1) / norm(self._first_direction.flatten(1)).reshape((-1, 1))
        self._first_direction = self._first_direction.reshape(originals.shape)

        # Run Attacks
        for _ in range(self._steps):
            self.best_advs = self._run(originals)
        return restore_type(self.best_advs)

    def _run(self, originals):
        # Get candidates
        candidates = self.get_candidates(originals)
        
        # Defin
        best_advs = self.best_advs.raw
        d_q = self.distance(originals, best_advs).raw
        last_d_q = copy.deepcopy(d_q)
        
        best_candidates = {str(i): -1 for i in range(len(originals))}
        n_candidates = {str(i): 0 for i in range(len(originals))}
        
        all_distances = [self.distance(originals, candidate).raw.cpu() for candidate in candidates]

        feedbacks = []
        for i in range(len(originals)):
            distances_i = []
            for c_i, candidate in enumerate(candidates):
                if (candidate[i] == 0).all():
                    continue
                distances_i.append((all_distances[c_i][i], c_i))
            n_candidates[str(i)] = len(distances_i)

            best_dist, best_cand = sorted(distances_i, key=lambda  x: x[0])[0]
            best_cand_adv = self._single_is_adversarial(candidates[best_cand][i], i)
            if d_q[i] > best_dist and best_cand_adv:
                feedbacks.append((last_d_q[i] - best_dist) / last_d_q[i])

                best_advs[i] = candidates[best_cand][i].raw
                d_q[i] = best_dist
                best_candidates[str(i)] = best_cand

            else:
                if d_q[i] > best_dist and not best_cand_adv:
                    print("Best cand not adversarial, Image {}, best_cand: {}, best_dist: {}".format(i, best_cand, best_dist))

                feedbacks.append(-1)

        return ep.astensor(best_advs)


    def get_candidates(self, originals):
        # find epsilon:
        qs, epsilons, function_evolution = self._get_epsilons(originals)
        if not self.with_interp:
            return qs

        # Start Interpolation
        alpha = ep.zeros(originals, len(originals)).raw
        distances = [self.distance(e, originals).raw for e in qs]
        epsilon_shape_added, qs_shape_added = [1] + list(epsilons.shape[1:]), [1] + list(qs.shape[1:])
        for i in range(len(originals)):
            n_epsilon = 3
            y = np.array([float(e[i].cpu()) for e in distances])
            x = np.array([float(e[i].raw.cpu()) for e in epsilons[0: n_epsilon]])
            a = get_minimum_interpolation(x, y)
            while a is None or abs(a) > max(abs(x)):
                n_epsilon += 1
                epsilons_i = sorted([e[i] for e in epsilons], key=lambda x: abs(x), reverse=True)
                new_epsilon = 2 * epsilons_i[0] #epsilons_i[0] + (epsilons_i[0] - epsilons_i[1])
                new_epsilon = float(new_epsilon.raw.cpu())
                new_q_i = function_evolution(new_epsilon)[i]
                new_q_i = self._single_vector_binary_search(originals[i], new_q_i, i, have_to_be_adv=False,  boost=True)
                if len(epsilons) < n_epsilon:
                    epsilons = ep.concatenate((epsilons, ep.zeros(epsilons, epsilon_shape_added)), axis = 0)
                    qs = ep.concatenate((qs, ep.zeros(qs, qs_shape_added)), axis = 0)

                epsilons, qs = epsilons.raw, qs.raw
                epsilons[n_epsilon - 1][i] = new_epsilon
                qs[n_epsilon - 1][i] = new_q_i.raw
                epsilons, qs = ep.astensors(epsilons, qs)

                y = np.array([float(self.distance(e[i].expand_dims(0), originals[i].expand_dims(0)).raw.cpu()) for e in qs[n_epsilon-3: n_epsilon]])
                x = np.array([float(e[i].raw.cpu()) for e in epsilons[n_epsilon-3: n_epsilon]])
                a = get_minimum_interpolation(x, y)

            alpha[i] = a

        alpha = ep.astensor(alpha)
        q_alpha = ep.zeros(originals, originals.shape)
        q_evolution_alpha = function_evolution(alpha)
        max_distance = self.distance(originals, self.best_advs).raw

        for i, q_a in enumerate(q_evolution_alpha):
            if any(abs(e - alpha[i]) < 0.5 for e in [e[i] for e in epsilons]):
                continue
            q_a = self._single_vector_binary_search(originals[i], q_a, i, have_to_be_adv=False, max_distance=max_distance[i], boost=True)
            q_alpha[i] = q_a.raw if q_a is not None else 0

        if self.device is not None:
            q_alpha = q_alpha.cuda(self.device)
        q_alpha = ep.astensor(q_alpha)

        qs = ep.concatenate((qs, q_alpha.expand_dims(0)), axis=0)

        return qs

    def _get_evolution_function(self, q, originals):
        d = np.array(self.distance(originals, q).raw.cpu())
        direction_1 = (q - originals).flatten(start=1) / self.distance(originals, q).reshape((-1, 1))
        direction_1 = direction_1.reshape(q.shape)
        return lambda theta, direction: (originals + add_step_in_circular_direction(direction_1, direction, d, theta)).clip(0, 1)

    def _get_best_alpha_direction(self, direction, index, function_evolution):
        theta_max = self.theta_max[index]
        iterator = []
        for i in range(0, self.T):
            coeff = 1 - i / self.T
            iterator.append(coeff * theta_max)
            iterator.append(- coeff * theta_max)

        for param in iterator:
            x = function_evolution(param, direction)[0]
            if self._single_is_adversarial(x, index):
                return param
        return None

    def _alpha_binary_search(self, direction, index, function_evolution):    
        # Upper --> not adversarial /  Lower --> adversarial
        lower = self._get_best_alpha_direction(direction, index, function_evolution)
        if lower is None:
            return None

        get_alpha = lambda theta: 1 - np.cos(theta * np.pi / 180)
        check_opposite = lower > 0 # if param< 0: abs(param) doesn't work
        
        if abs(lower) != self.theta_max[index]:
            sign = np.sign(lower)
            upper = lower + sign * self.theta_max[index] / 3
        else:
            # Find the correct range
            step = lower / 3
            upper = lower + step
            while self._single_is_adversarial(function_evolution(upper, direction)[0], index):
                lower = upper
                upper = lower + step
                
        step = 0
        while step < self.BS_max_iteration and abs(get_alpha(upper) - get_alpha(lower)) > self.BS_gamma: #upper - lower > thresholds:
            mid_bound = (upper + lower) / 2
            mid = function_evolution(mid_bound, direction)[0]

            mid_opp = function_evolution(-mid_bound, direction)[0]
            if self._single_is_adversarial(mid, index):
                lower = mid_bound
                if check_opposite:
                    check_opposite = self._single_is_adversarial(mid_opp, index)
            elif check_opposite and self._single_is_adversarial(mid_opp, index):
                lower = -mid_bound
                upper = -upper
                check_opposite = False
            else:
                upper = mid_bound
            step += 1
        return lower


    def _get_epsilons(self, originals):
        """
        Find the lowest epsilon to misclassified x following the direction: q of class 1 / q + eps*direction of class 0
        """
        count_fail = {i: 0 for i in range(len(originals))}

        final_directions = []
        qs = ep.zeros(self.best_advs, [3] + list(originals.shape)).raw
        epsilons = ep.zeros(self.best_advs, (3, len(originals))).raw
        qs[0] = self.best_advs.raw

        for i in range(len(originals)):
            # For each image, find the good theta in this direction or take antoehr direction

            function_evolution_i = self._get_evolution_function(self.best_advs[i].expand_dims(0), originals[i].expand_dims(0))
            if i not in self._directions_ortho:
                self._directions_ortho[i] = []

            best_param = None
            while best_param is None:
                ortho_with = [self._first_direction[i].expand_dims(0)] + self._directions_ortho[i]
                ortho_with = ep.concatenate(ortho_with, axis=0)
                t = ep.astensor(self._basis.get_vector(i, self._first_direction[i].shape, ortho_with)).expand_dims(0)

                self._directions_ortho[i].append(t)   
                self._directions_ortho[i] = self._directions_ortho[i][-self.last_direction_ortho:]
                if not self.with_BS_alpha:
                    best_param = self._get_best_alpha_direction(t, i, function_evolution_i)
                else:
                    best_param = self._alpha_binary_search(t, i, function_evolution_i)

                if best_param is None:
                    # Bad Direction
                    count_fail[i] += 1
                    self.theta_max[i] *= self.rho   # Malus

            qs[1][i] = function_evolution_i(best_param, t)[0].raw

            epsilons[1][i], epsilons[2][i] = best_param, best_param / 2
            self.theta_max[i] /= self.rho   # Bonus

            if self.with_interp:
                qs[2][i] = function_evolution_i(epsilons[2][i], t)[0].raw

            if self.with_BS:
                qs[1][i] = self._single_vector_binary_search(originals[i], ep.astensor(qs[1][i]), i, have_to_be_adv=False, boost=True).raw
                qs[2][i] = self._single_vector_binary_search(originals[i], ep.astensor(qs[2][i]), i, have_to_be_adv=False, boost=True).raw

            final_directions.append(t)

        final_directions = ep.concatenate(final_directions, axis=0)
        function_evolution = self._get_evolution_function(self.best_advs, originals)
        return ep.astensor(qs), ep.astensor(epsilons), lambda x: function_evolution(x, final_directions)


    def _single_get_misclassified(self, u, v, index, step = 0.1, max_distance=None):
        """
        For adv not adversarial, find an adversarial in the direction (adv - o) / d.
        Move from in the direction u to v, v will be adversarial.
        Args:
            u ([type]): image 1
            v ([type]): image 2
            index ([type]): index of the image
            step (float, optional): Step walk in this direction. Defaults to 0.1.
            max_distance ([type], optional): Stop if no adversarial found in max_distance direction. Defaults to None.

        Returns:
        """
        d = self.distance(u.expand_dims(0), v.expand_dims(0))
        direction = (u - v) / d

        while not self._single_is_adversarial(v, index):
            v = v + step * direction
            if max_distance is not None and self.distance(v.expand_dims(0), u.expand_dims(0)) > max_distance:
                return None
        return v - step * direction, v

    def _single_vector_binary_search(self, u, v, index_truth, max_steps=10, ratio_threshold=0.001, have_to_be_adv=False, max_distance=None, return_coef=False, boost=False):
        # Perform Binary search between u and v
        if not self.with_BS:
            return v if self._single_is_adversarial(v, index_truth) else None
        upper, lower = 1, 0

        if not self._single_is_adversarial(v, index_truth):
            if have_to_be_adv:
                return None 
            misclassified = self._single_get_misclassified(u, v, index_truth, step=0.2, max_distance=max_distance)
            if misclassified is None:
                return None
            else:
                u, v = misclassified

        d = self.distance(u.expand_dims(0), v.expand_dims(0))
        thresholds = d * self.BS_gamma

        if boost:
            boost_vec = 0.1 * u + 0.9 * v
            if self._single_is_adversarial(boost_vec, index_truth):
                v = boost_vec
            else:
                u = boost_vec

        step = 0
        while step < max_steps and (upper - lower) * d > thresholds: #upper - lower > thresholds:
            mid_bound = (upper + lower) / 2
            mid = (1 - mid_bound) * u + mid_bound * v
            if not self._single_is_adversarial(mid, index_truth):
                lower = mid_bound
            else:
                upper = mid_bound

            step += 1
        if return_coef:
            return (1 - upper) * u + upper * v, upper
        else:
            return (1 - upper) * u + upper * v

    def _single_is_adversarial(self, adv, truth_index):
        self._nqueries[truth_index] += 1
        decision_adv = self._model(adv.expand_dims(0))[0].argmax(0)
        is_adv = (decision_adv != self._basis._original_labels[truth_index])
        return is_adv


def add_step_in_circular_direction(direction1, direction2, r, degree):
    degree = get_raw(degree)
    if isinstance(degree, torch.Tensor) and degree.is_cuda:
        degree = degree.cpu()
    r = r * np.cos(np.array(degree) * np.pi / 180)
    results = []
    dir1_flat = direction1.flatten(1)
    dir2_flat = direction2.flatten(1)
    for i in range(len(direction1)):
        try:
            deg = degree[i]
        except:
            deg = degree
        r_i = np.cos(deg * np.pi/180) * dir1_flat[i] + np.sin(deg * np.pi/180) * dir2_flat[i]

        results.append((r[i] * r_i).expand_dims(0))
    
    results = ep.concatenate(results, axis=0)
    return results.reshape(direction1.shape)


def get_minimum_interpolation(x, y, prec=0.1):
    f = interpolate.interp1d(x, y, kind="quadratic", fill_value="extrapolate")
    a = (f(1) + f(-1)) / 2 - f(0)

    x_min, x_min_2 = sorted(x, key=lambda e: abs(e), reverse=True)[:2]
    step = x_min - x_min_2
    if a < 0:
        return None
    
    b = (f(1) - f(-1)) / 2
    x_min_interp = - b / (2 * a)
    if abs(x_min_interp) > abs(x_min + step):
        return None
    return x_min_interp
    