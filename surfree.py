import logging
import numpy as np
import torch
import copy
import random
import eagerpy as ep
import math

from scipy import fft

from ..distances import l2

from ..devutils import atleast_kd

from .blended_noise import LinearSearchBlendedUniformNoiseAttack

from .base import MinimizationAttack, get_criterion, get_is_adversarial


class SurFree(MinimizationAttack):

    distance = l2
    def __init__(
        self, 
        steps=100, 
        max_queries=10000,
        BS_gamma=0.01, 
        BS_max_iteration=10, 
        theta_max=30, 
        n_ortho=100, 
        rho=0.98,  
        T=3, 
        with_alpha_line_search=True, 
        with_distance_line_search=False, 
        with_interpolation=False):
        """[summary]

        Args:
            steps (int, optional): run steps. Defaults to 1000.
            max_queries (int, optional): stop running when each example require max_queries.
            BS_gamma ([type], optional): Binary Search Early Stop. Stop when precision is below BS_gamma. Defaults to 0.01.
            BS_max_iteration ([type], optional): Max iteration for . Defaults to 10.
            theta_max (int, optional): max theta watched to evaluate the direction. Defaults to 30.
            evolution (str, optional): Move in this direction. It can be linear or circular. Defaults to "circular".
            n_ortho (int, optional): Orthogonalize with the n last directions. Defaults to 100.
            rho (float, optional): Malus/Bonus factor given to the theta_max. Defaults to 0.98.
            T (int, optional): How many evaluation done to evaluated a direction. Defaults to 1.
            with_alpha_line_search (bool, optional): Activate Binary Search on Theta. Defaults to True.
            with_distance_line_search (bool, optional): Activate Binary Search between adversarial and x_o. Defaults to False.
            with_interpolation (bool, optional): Activate Interpolation. Defaults to False.
        """
        # Attack Parameters
        self._BS_gamma = BS_gamma
        self._BS_max_iteration = BS_max_iteration

        self._steps = steps
        self._max_queries = max_queries
        self.best_advs = None
        self._theta_max = theta_max
        self.rho = rho
        self.T = T
        assert self.rho <= 1 and self.rho > 0

        # Add or remove some parts of the attack
        self.with_alpha_line_search = with_alpha_line_search
        self.with_distance_line_search = with_distance_line_search
        self.with_interpolation = with_interpolation
        if self.with_interpolation and not self.with_distance_line_search:
            Warning("It's higly recommended to use Interpolation with distance line search.")

        # Data saved during attack
        self.n_ortho = n_ortho
        self._directions_ortho = {}
        self._nqueries = {}

    def get_nqueries(self):
        return self._nqueries

    def run(self, model, inputs, criterion, early_stop=None, starting_points=None, basis_params=None, init_attack_step=10):
        self._nqueries = {i: 0 for i in range(len(inputs))}

        originals, restore_type = ep.astensor_(inputs)

        self._set_cos_sin_function(originals.raw)
        self.theta_max = ep.ones(inputs, len(inputs)) * self._theta_max
        criterion = get_criterion(criterion)
        self._criterion_is_adversarial = get_is_adversarial(criterion, model)

        # Get Starting Point
        if starting_points is not None:
            best_advs = starting_points
        elif starting_points is None:
            init_attack = LinearSearchBlendedUniformNoiseAttack(steps=init_attack_step)
            best_advs = init_attack.run(model, originals, criterion)
        else:
            raise ValueError("starting_points {} doesn't exist.".format(starting_points))

        assert self._is_adversarial(best_advs).all()
        # Initialize the direction orthogonalized with the first direction
        fd = best_advs - originals
        norm = ep.norms.l2(fd.flatten(1), axis=1)
        fd = fd / atleast_kd(norm, fd.ndim)
        self._directions_ortho = {i: v.expand_dims(0) for i, v in enumerate(fd)}


        # Load Basis
        if basis_params is None:
            basis_params = {}
        self._basis = Basis(originals, **basis_params)

        for _ in range(self._steps):
            # Get candidates. Shape: (n_candidates, batch_size, image_size)
            candidates = self._get_candidates(originals, best_advs)
            candidates = candidates.transpose((1, 0, 2, 3, 4))

            
            best_candidates = ep.zeros_like(best_advs).raw
            for i, o in enumerate(originals):
                o_repeated = ep.concatenate([o.expand_dims(0)] * len(candidates[i]), axis=0)
                index = ep.argmax(self.distance(o_repeated, candidates[i])).raw
                best_candidates[i] = candidates[i][index].raw

            is_success = self.distance(best_candidates, originals) < self.distance(best_advs, originals)
            best_advs = ep.where(atleast_kd(is_success, best_candidates.ndim), ep.astensor(best_candidates), best_advs)

            if all(v > self._max_queries for v in self._nqueries.values()):
                print("Max queries attained for all the images.")
                break

        return restore_type(best_advs)

    def _is_adversarial(self, perturbed):
        # Count the queries made for each image
        # Count if the vector is different from the null vector
        for i, p in enumerate(perturbed):
            if not (p == 0).all():
                self._nqueries[i] += 1
        return self._criterion_is_adversarial(perturbed)

    def _get_candidates(self, originals, best_advs):
        """
        Find the lowest epsilon to misclassified x following the direction: q of class 1 / q + eps*direction of class 0
        """
        epsilons = ep.zeros(originals, len(originals))
        direction_2 = ep.zeros_like(originals)
        while (epsilons == 0).any():
            direction_2 = ep.where(
                atleast_kd(epsilons == 0, direction_2.ndim),
                self._basis.get_vector(self._basis.get_vector(self._directions_ortho)),
                direction_2
            )
            
            for i, eps_i in enumerate(epsilons):
                if eps_i == 0:
                    # Concatenate the first directions and the last directions generated
                    self._directions_ortho[i] = ep.concatenate((
                        self._directions_ortho[i][:1],
                        self._directions_ortho[i][1 + len(self._directions_ortho[i]) - self.n_ortho:], 
                        direction_2[i].expand_dims(0)), axis=0)
                        
            function_evolution = self._get_evolution_function(originals, best_advs, direction_2)
            new_epsilons = self._get_best_theta(originals, function_evolution, epsilons)

            self.theta_max = ep.where(new_epsilons == 0, self.theta_max * self.rho, self.theta_max)
            self.theta_max = ep.where((new_epsilons != 0) * (epsilons == 0), self.theta_max / self.rho, self.theta_max)
            epsilons = new_epsilons

        epsilons = epsilons.expand_dims(0)
        if self.with_interpolation:
            epsilons =  ep.concatenate((epsilons, epsilons[0] / 2), axis=0)

        candidates = [function_evolution(eps).expand_dims(0) for eps in epsilons]
        candidates = ep.concatenate(candidates, axis=0)

        if self.with_interpolation:
            d = self.distance(best_advs, originals)
            delta = self.distance(self._binary_search(self._is_adversarial, originals, candidates[1],  boost=True), originals)
            theta_star = epsilons[0]

            num = theta_star * (4 * delta - d * (self._cos(theta_star.raw) + 3))
            den = 4 * (2 * delta - d * (self._cos(theta_star.raw) + 1))

            theta_hat = num / den
            q_interp = function_evolution(theta_hat)
            if self.with_distance_line_search:
                q_interp = self._binary_search(self._is_adversarial, originals, q_interp,  boost=True)
            candidates = ep.concatenate((candidates, q_interp.expand_dims(0)), axis=0)

        return candidates

    def _get_evolution_function(self, originals, best_advs, direction_2):
        distances = self.distance(best_advs, originals)
        direction_1 = (best_advs - originals).flatten(start=1) / distances.reshape((-1, 1))
        direction_1 = direction_1.reshape(originals.shape)
        return lambda theta: (originals + self._add_step_in_circular_direction(direction_1, direction_2, distances, theta)).clip(0, 1)

    def _get_best_theta(self, originals, function_evolution, best_params=None):
        iterator = ep.zeros(originals, (2 * self.T, len(self.theta_max))).raw
        for i in range(0, self.T):
            c = 1 - (i / self.T)
            for j in range(len(self.theta_max)):
                iterator[2* i, j] = c * self.theta_max[j].raw
                iterator[2 * i + 1, j] = - iterator[2* i, j]

        if best_params is None:
            best_params = ep.zeros_like(self.theta_max)
        mask = (best_params != 0)
        for i,  params in enumerate(ep.astensor(iterator)):
            params = mask * best_params + mask.logical_not() * params
            x = function_evolution(params)
            mask = self._is_adversarial(x)
            best_params = mask * params + mask.logical_not() * best_params

            if mask.all():
                break
        if (best_params == 0).all():
            return best_params
        elif self.with_alpha_line_search:
            return self._alpha_binary_search(function_evolution, best_params, mask)
        else:
            return best_params


    def _alpha_binary_search(self, function_evolution, lower, mask):    
        # Upper --> not adversarial /  Lower --> adversarial
        get_alpha = lambda theta: 1 - self._cos(theta.raw * np.pi / 180)
        check_opposite = lower > 0 # if param < 0: abs(param) doesn't work
        
        # Get the upper range
        upper = ep.where(
            ep.logical_and(abs(lower) != self.theta_max, mask), 
            lower + ep.sign(lower) * self.theta_max / self.T,
            ep.zeros_like(lower)
            )

        mask_upper = ep.logical_and(upper == 0, mask)
        while mask_upper.any():
            # Find the correct lower/upper range
            upper = ep.where(
                mask_upper,
                lower + ep.sign(lower) * self.theta_max / self.T,
                upper
            )
            x = function_evolution(upper)

            mask_upper = mask_upper * self._is_adversarial(x)
            lower = ep.where(mask_upper, upper, lower) 

        step = 0
        while step < self._BS_max_iteration and (abs(get_alpha(upper) - get_alpha(lower)) > self._BS_gamma).any(): 
            mid_bound = (upper + lower) / 2
            mid = function_evolution(mid_bound)
            is_adv = self._is_adversarial(mid)

            mid_opp = ep.where(
                atleast_kd(ep.astensor(check_opposite), mid.ndim),
                function_evolution(-mid_bound),
                ep.zeros_like(mid)
            )
            is_adv_opp = self._is_adversarial(mid_opp)

            lower = ep.where(mask * is_adv, mid_bound, lower)

            lower = ep.where(mask * is_adv.logical_not() * check_opposite * is_adv_opp, -mid_bound, lower)
            upper = ep.where(mask * is_adv.logical_not() * check_opposite * is_adv_opp, - upper, upper)

            upper = ep.where(mask * abs(lower) != abs(mid_bound), mid_bound, upper)

            check_opposite = mask * check_opposite * is_adv_opp * (lower > 0)

            step += 1
        return ep.astensor(lower)

    def _binary_search(self, is_adversarial, originals: ep.Tensor, perturbed: ep.Tensor, boost=False) -> ep.Tensor:
        # Choose upper thresholds in binary search based on constraint.
        highs = ep.ones(perturbed, len(perturbed))
        d = np.prod(perturbed.shape[1:])
        thresholds = self._BS_gamma / (d * math.sqrt(d))
        lows = ep.zeros_like(highs)

        # Boost Binary search
        if boost:
            boost_vec = 0.1 * originals + 0.9 * perturbed
            is_advs = is_adversarial(boost_vec)
            is_advs = atleast_kd(is_advs, originals.ndim)
            originals = ep.where(is_advs.logical_not(), boost_vec, originals)
            perturbed = ep.where(is_advs, boost_vec, perturbed)

        # use this variable to check when mids stays constant and the BS has converged
        old_mids = highs
        iteration = 0
        while ep.any(highs - lows > thresholds) and iteration < self._BS_max_iteration:
            iteration += 1
            mids = (lows + highs) / 2
            mids_perturbed = self._project(originals, perturbed, mids)
            is_adversarial_ = is_adversarial(mids_perturbed)

            highs = ep.where(is_adversarial_, mids, highs)
            lows = ep.where(is_adversarial_, lows, mids)

            # check of there is no more progress due to numerical imprecision
            reached_numerical_precision = (old_mids == mids).all()
            old_mids = mids
            if reached_numerical_precision:
                break
        
        results = self._project(originals, perturbed, highs)
        return results

    def _project(
            self, originals: ep.Tensor, perturbed: ep.Tensor, epsilons: ep.Tensor
        ) -> ep.Tensor:
        epsilons = atleast_kd(epsilons, originals.ndim)
        return (1.0 - epsilons) * originals + epsilons * perturbed

    def _add_step_in_circular_direction(self, direction1, direction2, r, degree):
        degree = atleast_kd(degree, direction1.ndim).raw
        r = atleast_kd(r, direction1.ndim)
        results = self._cos(degree * np.pi/180) * direction1 + self._sin(degree * np.pi/180) * direction2
        results = results * r * self._cos(degree * np.pi / 180)
        
        return results.reshape(direction1.shape)

    def _set_cos_sin_function(self, v):
        if isinstance(v, torch.Tensor):
            self._cos, self._sin = torch.cos, torch.sin
        elif isinstance(v, np.array):
            self._cos, self._sin = np.cos, np.sin
        else:
            raise ValueError("Cos and sin functions, not available for this instances.")


class Basis:
    def __init__(self, originals, random_noise="normal", basis_type="dct", *args, **kwargs):
        """
        Args:
            random_noise (str, optional): When basis is created, a noise will be added.This noise can be normal or uniform. Defaults to "normal".
            basis_type (str, optional): Type of the basis: DCT, Random, Genetic,. Defaults to "random".
è            device (int, optional): [description]. Defaults to -1.
            args, kwargs: In args and kwargs, there is the basis params:
                    * Random: No parameters                    
                    * DCT:
                            * function (tanh / constant / linear): function applied on the dct
                            * alpha
                            * beta
                            * lambda
                            * frequence_range: integers or float
                            * min_dct_value
                            * dct_type: 8x8 or full
        """
        self._originals = originals
        self._direction_shape = originals.shape[1:]
        self.basis_type = basis_type

        self._load_params_type(*args, **kwargs)

        assert random_noise in ["normal", "uniform"]
        self.random_noise = random_noise

    def get_vector(self, ortho_with=None, bounds=(0, 1)):
        if ortho_with is None:
            ortho_with = {i: None for i in range(len(self._originals))}

        vectors = [
            self.get_vector_i(i, ortho_with[i], bounds).expand_dims(0)
            for i in range(len(self._originals))
        ]
        return ep.concatenate(vectors, axis=0)
    
    def get_vector_i(self, index, ortho_with=None, bounds=(0, 1)):
        r = getattr(self, "_get_vector_i_" + self.basis_type)(index, bounds)
        if ortho_with is not None:
            r_repeated = ep.concatenate([r.expand_dims(0)] * len(ortho_with), axis=0)
            
            #inner product
            gs_coeff = (ortho_with * r_repeated).flatten(1).sum(1)
            proj = atleast_kd(gs_coeff, ortho_with.ndim) * ortho_with
            r = r - proj.sum(0)
        return r / ep.norms.l2(r)

    def _get_vector_i_dct(self, index, bounds):
        r_np = np.zeros(self._direction_shape)
        for channel, dct_channel in enumerate(self.dcts[index]):
            probs = np.random.randint(-2, 1, dct_channel.shape) + 1
            r_np[channel] = dct_channel * probs
        r_np = idct2_8_8(r_np) + self._beta * (2 * np.random.rand(*r_np.shape) - 1)
        return ep.from_numpy(self._originals, r_np.astype("float32"))

    def _get_vector_i_random(self, index, bounds):
        r = ep.zeros(self._originals, self._direction_shape)
        r = getattr(ep, self.random_noise)(r, r.shape, *bounds)
        return ep.astensor(r)

    def _load_params_type(self, **kwargs):
        if not hasattr(self, "_get_vector_i_" + self.basis_type):
            raise ValueError("Basis {} doesn't exist.".format(self.basis_type))

        if self.basis_type == "dct":
            self._beta = kwargs["beta"] if "beta" in kwargs else 0
            frequence_range = kwargs["frequence_range"] if "frequence_range" in kwargs else (0, 1)

            if "dct_type" not in kwargs or kwargs["dct_type"] == "8x8":
                dct_mask = get_zig_zag_mask(frequence_range, (8, 8))
                self.dcts = np.array([dct2_8_8(np.array(image.raw.cpu()), dct_mask) for image in self._originals])
            elif kwargs["dct_type"] == "full":
                dct_mask = get_zig_zag_mask(frequence_range, self._originals.shape[2:])
                self.dcts = np.array([dct2_full(np.array(image.raw.cpu()), dct_mask) for image in self._originals])
            else:
                raise ValueError("DCT {} doesn't exist.".format(kwargs["dct_type"]))

            min_value_dct = kwargs["min_value_dct"] if "min_value_dct" in kwargs else 0
            self.dcts *= (abs(self.dcts) > min_value_dct).astype(int)
            
            if "function" not in kwargs or kwargs["function"] == "tanh":
                lambd = kwargs["lambda"] if "lambda" in kwargs else 1
                f = lambda x: np.tanh(lambd * x)
            elif kwargs["function"] == "identity":
                f = lambda x: x
            elif kwargs["function"] == "constant":
                f = lambda x: (abs(x) > 0).astype(int)
            else:
                raise ValueError("Function given for DCT is incorrect.")
            
            self.dcts = f(self.dcts)


###########################
# DCT Functions
###########################


def dct2(a):
    return fft.dct(fft.dct(a, axis=0, norm='ortho' ), axis=1, norm='ortho')

def idct2(a):
    return fft.idct(fft.idct(a, axis=0 , norm='ortho'), axis=1 , norm='ortho')

def dct2_8_8(image, mask=None):
    if mask is None:
        mask = np.ones((8, 8))
    if mask.shape != (8, 8):
        raise ValueError("Mask have to be with a size of (8, 8)")

    imsize = image.shape
    dct = np.zeros(imsize)
    
    for channel in range(imsize[0]):
        for i in np.r_[:imsize[1]:8]:
            for j in np.r_[:imsize[2]:8]:
                dct_i_j = dct2(image[channel, i:(i+8),j:(j+8)]) 
                dct[channel, i:(i+8),j:(j+8)] = dct_i_j * mask[:dct_i_j.shape[0], :dct_i_j.shape[1]]
    return dct

def idct2_8_8(dct):
    im_dct = np.zeros(dct.shape)
    
    for channel in range(dct.shape[0]):
        for i in np.r_[:dct.shape[1]:8]:
            for j in np.r_[:dct.shape[2]:8]:
                im_dct[channel, i:(i+8),j:(j+8)] = idct2(dct[channel, i:(i+8),j:(j+8)] )
    return im_dct

def dct2_full(image, mask=None):
    if mask is None:
        mask = np.ones(image.shape[-2:])

    imsize = image.shape
    dct = np.zeros(imsize)
    
    for channel in range(imsize[0]):
            dct_i_j = dct2(image[channel] ) 
            dct[channel] = dct_i_j * mask
    return dct

def idct2_full(dct):
    im_dct = np.zeros(dct.shape)
    
    for channel in range(dct.shape[0]):
        im_dct[channel] = idct2(dct[channel])
    return im_dct

def get_zig_zag_mask(frequence_range, mask_shape=(8, 8)):
    mask = np.zeros(mask_shape)
    s = 0
    total_component = sum(mask.flatten().shape)
    
    if isinstance(frequence_range[1], float) and frequence_range[1] <= 1:
        n_coeff = int(total_component * frequence_range[1])
    else:
        n_coeff = int(frequence_range[1])

    if isinstance(frequence_range[0], float) and frequence_range[0] <= 1:
        min_coeff = int(total_component * frequence_range[0])
    else:
        min_coeff = int(frequence_range[0])
    
    while n_coeff > 0:
        for i in range(min(s + 1, mask_shape[0])):
            for j in range(min(s + 1, mask_shape[1])):
                if i + j == s:
                    if min_coeff > 0:
                        min_coeff -= 1
                        continue

                    if s % 2:
                        mask[i, j] = 1
                    else:
                        mask[j, i] = 1
                    n_coeff -= 1
                    if n_coeff == 0:
                        return mask
        s += 1
    return mask
