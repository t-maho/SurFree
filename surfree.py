import logging
import numpy as np
import torch
import copy
import random
import eagerpy as ep
import math

from scipy import fft

from ..models import Model

from ..criteria import Criterion

from ..distances import l2

from ..devutils import atleast_kd

from .blended_noise import LinearSearchBlendedUniformNoiseAttack

from .base import MinimizationAttack, get_criterion, get_is_adversarial
from .base import T

from typing import Callable, Union, Optional, Tuple, List, Any, Dict


class SurFree(MinimizationAttack):
    distance = l2
    def __init__(
        self, 
        steps: int = 100, 
        max_queries: int = 10000,
        BS_gamma: float = 0.01, 
        BS_max_iteration: int = 10, 
        theta_max: float = 30, 
        n_ortho: int = 100, 
        rho: float = 0.98,  
        T: int = 3, 
        with_alpha_line_search: bool = True, 
        with_distance_line_search: bool = False, 
        with_interpolation: bool = False):
        """
        Args:
            steps (int, optional): run steps. Defaults to 1000.
            max_queries (int, optional): stop running when each example require max_queries.
            BS_gamma ([type], optional): Binary Search Early Stop. Stop when precision is below BS_gamma. Defaults to 0.01.
            BS_max_iteration ([type], optional): Max iteration for . Defaults to 10.
            theta_max (int, optional): max theta watched to evaluate the direction. Defaults to 30.
            evolution (str, optional): Move in this direction. It can be linear or circular. Defaults to "circular".
            n_ortho (int, optional): Orthogonalize with the n last directions. Defaults to 100.
            rho (float, optional): Bonus/Malus factor given to the theta_max for each direction tried. Defaults to 0.98.
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
        self._directions_ortho: Dict[int, ep.Tensor] = {}
        self._nqueries: Dict[int, int] = {}
        self._basis: Basis = None

    def get_nqueries(self) -> Dict:
        return self._nqueries

    def run(
        self,
        model: Model,
        inputs: T,
        criterion: Union[Criterion, T],
        *,
        early_stop: Optional[float] = None,
        starting_points: Optional[ep.Tensor] = None,
        **kwargs: Any,
    ) -> T:
        originals, restore_type = ep.astensor_(inputs)

        self._nqueries = {i: 0 for i in range(len(originals))}
        self._set_cos_sin_function(originals)
        self.theta_max = ep.ones(originals, len(originals)) * self._theta_max
        criterion = get_criterion(criterion)
        self._criterion_is_adversarial = get_is_adversarial(criterion, model)

        # Get Starting Point
        if starting_points is not None:
            best_advs = starting_points
        elif starting_points is None:
            init_attack: MinimizationAttack = LinearSearchBlendedUniformNoiseAttack(steps=50)
            best_advs = init_attack.run(model, originals, criterion, early_stop=early_stop)
        else:
            raise ValueError("starting_points {} doesn't exist.".format(starting_points))

        assert self._is_adversarial(best_advs).all()

        # Initialize the direction orthogonalized with the first direction
        fd = best_advs - originals
        norm = ep.norms.l2(fd.flatten(1), axis=1)
        fd = fd / atleast_kd(norm, fd.ndim)
        self._directions_ortho = {i: v.expand_dims(0) for i, v in enumerate(fd)}

        # Load Basis
        if "basis_params" in kwargs:
            self._basis = Basis(originals, **kwargs["basis_params"])
        else:
            self._basis = Basis(originals)

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
            print(self._nqueries)
            print(l2(best_advs, originals))

            if all(v > self._max_queries for v in self._nqueries.values()):
                print("Max queries attained for all the images.")
                break

        return restore_type(best_advs)

    def _is_adversarial(self, perturbed: ep.Tensor) -> ep.Tensor:
        # Count the queries made for each image
        # Count if the vector is different from the null vector
        for i, p in enumerate(perturbed):
            if not (p == 0).all():
                self._nqueries[i] += 1
        return self._criterion_is_adversarial(perturbed)

    def _get_candidates(self, originals: ep.Tensor, best_advs: ep.Tensor) -> ep.Tensor:
        """
        Find the lowest epsilon to misclassified x following the direction: q of class 1 / q + eps*direction of class 0
        """
        epsilons = ep.zeros(originals, len(originals))
        direction_2 = ep.zeros_like(originals)
        while (epsilons == 0).any():
            # if epsilon ==0, we are still searching a good direction
            direction_2 = ep.where(
                atleast_kd(epsilons == 0, direction_2.ndim),
                self._basis.get_vector(self._directions_ortho),
                direction_2
            )
            
            for i, eps_i in enumerate(epsilons):
                if eps_i == 0:
                    self._directions_ortho[i] = ep.concatenate((self._directions_ortho[i], direction_2[i].expand_dims(0)), axis=0)
                    if len(self._directions_ortho[i]) > self.n_ortho + 1:
                        self._directions_ortho[i] = ep.concatenate((self._directions_ortho[i][:1], self._directions_ortho[i][self.n_ortho:]))
                        
            function_evolution = self._get_evolution_function(originals, best_advs, direction_2)
            new_epsilons = self._get_best_theta(function_evolution, epsilons)

            self.theta_max = ep.where(new_epsilons == 0, self.theta_max * self.rho, self.theta_max)
            self.theta_max = ep.where((new_epsilons != 0) * (epsilons == 0), self.theta_max / self.rho, self.theta_max)
            epsilons = new_epsilons

        function_evolution = self._get_evolution_function(originals, best_advs, direction_2)
        if self.with_alpha_line_search:
            epsilons = self._alpha_binary_search(function_evolution, epsilons)

        epsilons = epsilons.expand_dims(0)
        if self.with_interpolation:
            epsilons =  ep.concatenate((epsilons, epsilons[0] / 2), axis=0)

        candidates = ep.concatenate([function_evolution(eps).expand_dims(0) for eps in epsilons], axis=0)

        if self.with_interpolation:
            d = self.distance(best_advs, originals)
            delta = self.distance(self._binary_search(originals, candidates[1],  boost=True), originals)
            theta_star = epsilons[0]

            num = theta_star * (4 * delta - d * (self._cos(theta_star.raw) + 3))
            den = 4 * (2 * delta - d * (self._cos(theta_star.raw) + 1))

            theta_hat = num / den
            q_interp = function_evolution(theta_hat)
            if self.with_distance_line_search:
                q_interp = self._binary_search(originals, q_interp,  boost=True)
            candidates = ep.concatenate((candidates, q_interp.expand_dims(0)), axis=0)

        return candidates

    def _get_evolution_function(self, originals: ep.Tensor, best_advs: ep.Tensor, direction2: ep.Tensor) -> Callable[[ep.Tensor], ep.Tensor]:
        distances = self.distance(best_advs, originals)
        direction1 = (best_advs - originals).flatten(start=1) / distances.reshape((-1, 1))
        direction1 = direction1.reshape(originals.shape)
        distances = atleast_kd(distances, direction1.ndim)

        def _add_step_in_circular_direction(degree: ep.Tensor) -> ep.Tensor:
            degree = atleast_kd(degree, direction1.ndim).raw * np.pi / 180
            results = self._cos(degree) * direction1 + self._sin(degree) * direction2
            results = results * distances * self._cos(degree)
            return (originals + ep.astensor(results)).clip(0, 1)

        return _add_step_in_circular_direction

    def _get_best_theta(
            self, 
            function_evolution: Callable[[ep.Tensor], ep.Tensor], 
            best_params: ep.Tensor) -> ep.Tensor:
        v_type = function_evolution(best_params)
        coefficients = ep.zeros(v_type, 2 * self.T).raw
        for i in range(0, self.T):
            coefficients[2* i] = 1 - (i / self.T)
            coefficients[2 * i + 1] = - coefficients[2* i]

        for i,  coeff in enumerate(coefficients):
            params = coeff * self.theta_max
            x_evol = function_evolution(params)
            x = ep.where(
                atleast_kd(best_params == 0, v_type.ndim), 
                x_evol, 
                ep.zeros_like(v_type))

            is_advs = self._is_adversarial(x)

            best_params = ep.where(
                (best_params == 0) * is_advs,
                params,
                best_params
            )
            if (best_params != 0).all():
                break
        
        return best_params  

    def _alpha_binary_search(
            self, 
            function_evolution: Callable[[ep.Tensor], ep.Tensor], 
            lower: ep.Tensor) -> ep.Tensor:    
        # Upper --> not adversarial /  Lower --> adversarial
        v_type = function_evolution(lower)
        def get_alpha(theta: ep.Tensor) -> ep.Tensor:
            return 1 - ep.astensor(self._cos(theta.raw * np.pi / 180))

        check_opposite = lower > 0 # if param < 0: abs(param) doesn't work
        
        # Get the upper range
        upper = ep.where(
            abs(lower) != self.theta_max, 
            lower + ep.sign(lower) * self.theta_max / self.T,
            ep.zeros_like(lower)
            )

        mask_upper = (upper == 0)
        while mask_upper.any():
            # Find the correct lower/upper range
            # if True in mask_upper, the range haven't been found
            new_upper = lower + ep.sign(lower) * self.theta_max / self.T
            potential_x = function_evolution(new_upper)
            x = ep.where(
                atleast_kd(mask_upper, potential_x.ndim),
                potential_x,
                ep.zeros_like(potential_x)
            )

            is_advs =  self._is_adversarial(x)
            lower = ep.where(ep.logical_and(mask_upper, is_advs), new_upper, lower) 
            upper = ep.where(ep.logical_and(mask_upper, is_advs.logical_not()), new_upper, upper) 
            mask_upper = mask_upper * is_advs

        step = 0
        over_gamma = abs(get_alpha(upper) - get_alpha(lower)) > self._BS_gamma
        while step < self._BS_max_iteration and over_gamma.any(): 
            mid_bound = (upper + lower) / 2
            mid = ep.where(
                atleast_kd(ep.logical_and(mid_bound != 0, over_gamma), v_type.ndim),
                function_evolution(mid_bound),
                ep.zeros_like(v_type)
            )
            is_adv = self._is_adversarial(mid)

            mid_opp = ep.where(
                atleast_kd(ep.logical_and(ep.astensor(check_opposite), over_gamma), mid.ndim),
                function_evolution(-mid_bound),
                ep.zeros_like(mid)
            )
            is_adv_opp = self._is_adversarial(mid_opp)

            lower = ep.where(over_gamma * is_adv, mid_bound, lower)
            lower = ep.where(over_gamma * is_adv.logical_not() * check_opposite * is_adv_opp, -mid_bound, lower)
            upper = ep.where(over_gamma * is_adv.logical_not() * check_opposite * is_adv_opp, - upper, upper)
            upper = ep.where(over_gamma * (abs(lower) != abs(mid_bound)), mid_bound, upper)

            check_opposite = over_gamma * check_opposite * is_adv_opp * (lower > 0)
            over_gamma = abs(get_alpha(upper) - get_alpha(lower)) > self._BS_gamma

            step += 1
        return ep.astensor(lower)

    def _binary_search(self, originals: ep.Tensor, perturbed: ep.Tensor, boost: Optional[bool] = False) -> ep.Tensor:
        # Choose upper thresholds in binary search based on constraint.
        highs = ep.ones(perturbed, len(perturbed))
        d = np.prod(perturbed.shape[1:])
        thresholds = self._BS_gamma / (d * math.sqrt(d))
        lows = ep.zeros_like(highs)

        # Boost Binary search
        if boost:
            boost_vec = 0.1 * originals + 0.9 * perturbed
            is_advs = self._is_adversarial(boost_vec)
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
            is_adversarial_ = self._is_adversarial(mids_perturbed)

            highs = ep.where(is_adversarial_, mids, highs)
            lows = ep.where(is_adversarial_, lows, mids)

            # check of there is no more progress due to numerical imprecision
            reached_numerical_precision = (old_mids == mids).all()
            old_mids = mids
            if reached_numerical_precision:
                break
        
        results = self._project(originals, perturbed, highs)
        return results

    def _project(self, originals: ep.Tensor, perturbed: ep.Tensor, epsilons: ep.Tensor) -> ep.Tensor:
        epsilons = atleast_kd(epsilons, originals.ndim)
        return (1.0 - epsilons) * originals + epsilons * perturbed

    def _set_cos_sin_function(self, v: ep.Tensor) -> None:
        if isinstance(v.raw, torch.Tensor):
            self._cos, self._sin = torch.cos, torch.sin
        elif isinstance(v.raw, np.array):
            self._cos, self._sin = np.cos, np.sin
        else:
            raise ValueError("Cos and sin functions, not available for this instances.")


class Basis:
    def __init__(self, originals: ep.Tensor, random_noise: str = "normal", basis_type: str = "dct", **kwargs : Any):
        """
        Args:
            random_noise (str, optional): When basis is created, a noise will be added.This noise can be normal or 
                                          uniform. Defaults to "normal".
            basis_type (str, optional): Type of the basis: DCT, Random, Genetic,. Defaults to "random".
            device (int, optional): [description]. Defaults to -1.
            args, kwargs: In args and kwargs, there is the basis params:
                    * Random: No parameters                    
                    * DCT:
                            * function (tanh / constant / linear): function applied on the dct
                            * beta
                            * gamma
                            * frequence_range: tuple of 2 float
                            * dct_type: 8x8 or full
        """
        self._originals = originals
        self.basis_type = basis_type

        self._load_params(**kwargs)

        assert random_noise in ["normal", "uniform"]
        self.random_noise = random_noise

    def get_vector(self, ortho_with: Optional[Dict] = None, bounds: Tuple[float, float] = (0, 1)) -> ep.Tensor:
        if ortho_with is None:
            ortho_with = {i: None for i in range(len(self._originals))}
        r: ep.Tensor = getattr(self, "_get_vector_" + self.basis_type)()

        vectors = [
            self._gram_schmidt(r[i], ortho_with[i]).expand_dims(0)
            for i in ortho_with
        ]
        return ep.concatenate(vectors, axis=0)

    def _gram_schmidt(self, v: ep.Tensor, ortho_with: ep.Tensor):
        v_repeated = ep.concatenate([v.expand_dims(0)] * len(ortho_with), axis=0)
        
        #inner product
        gs_coeff = (ortho_with * v_repeated).flatten(1).sum(1)
        proj = atleast_kd(gs_coeff, ortho_with.ndim) * ortho_with
        v = v - proj.sum(0)
        return v / ep.norms.l2(v)

    def _get_vector_dct(self) -> ep.Tensor:
        probs = np.random.randint(-2, 1, self.dcts.shape) + 1
        r_np = self.dcts * probs
        r_np = ep.from_numpy(self._originals, self._inverse_dct(r_np).astype("float32"))
        return r_np + ep.normal(self._originals, r_np.shape, stddev=self._beta)

    def _get_vector_random(self) -> ep.Tensor:
        r = ep.zeros_like(self._originals)
        r = getattr(ep, self.random_noise)(r, r.shape, 0, 1)
        return ep.astensor(r)

    def _load_params(
            self, 
            beta: float = 0,
            frequence_range: Tuple[float, float] = (0, 1),
            dct_type: str = "8x8",
            function: str = "tanh",
            gamma: float = 1
            ) -> None:
        if not hasattr(self, "_get_vector_" + self.basis_type):
            raise ValueError("Basis {} doesn't exist.".format(self.basis_type))

        if self.basis_type == "dct":
            self._beta = beta
            if dct_type == "8x8":
                mask_size = (8, 8) 
                dct_function = dct2_8_8
                self._inverse_dct = idct2_8_8
            elif dct_type == "full":
                mask_size = self._originals.shape[-2:]
                dct_function = lambda x, mask: dct2(x) * mask
                self._inverse_dct = idct2
            else:
                raise ValueError("DCT {} doesn't exist.".format(dct_type))
            
            dct_mask = get_zig_zag_mask(frequence_range, mask_size)
            print(dct_mask)
            imsize = self._originals.shape
            dct_mask = dct_mask.reshape((1, 1, dct_mask.shape[0], dct_mask.shape[1])).repeat(imsize[0], 0).repeat(imsize[1], 1)

            self.dcts = np.array(dct_function(np.array(self._originals.raw.cpu()), dct_mask))
        
            def get_function(function: str, gamma: float) -> Callable:
                if function == "tanh":
                    return lambda x: np.tanh(gamma * x)
                elif function == "identity":
                    return lambda x: x
                elif function == "constant":
                    return lambda x: (abs(x) > 0).astype(int)
                else:
                    raise ValueError("Function given for DCT is incorrect.")

            self.dcts = get_function(function, gamma)(self.dcts)


###########################
# DCT Functions
###########################


def dct2(a: Any) -> Any:
    return fft.dct(fft.dct(a, axis=2, norm='ortho' ), axis=3, norm='ortho')


def idct2(a: Any) -> Any:
    return fft.idct(fft.idct(a, axis=2, norm='ortho'), axis=3, norm='ortho')


def dct2_8_8(image: Any, mask: Any) -> Any:
    assert mask.shape[-2:] == (8, 8)

    imsize = image.shape
    dct = np.zeros(imsize)
    for i in np.r_[:imsize[2]:8]:
        for j in np.r_[:imsize[3]:8]:
            dct_i_j = dct2(image[:, :, i:(i+8),j:(j+8)]) 
            dct[:, :, i:(i+8),j:(j+8)] = dct_i_j * mask#[:dct_i_j.shape[0], :dct_i_j.shape[1]]
    return dct


def idct2_8_8(dct: Any) -> Any:
    im_dct = np.zeros(dct.shape)
    for i in np.r_[:dct.shape[2]:8]:
        for j in np.r_[:dct.shape[3]:8]:
            im_dct[:, :, i:(i+8),j:(j+8)] = idct2(dct[:, :, i:(i+8),j:(j+8)])
    return im_dct


def get_zig_zag_mask(frequence_range: Tuple[float, float], mask_shape: Tuple[int, int] = (8, 8)) -> Any:
    mask = np.zeros(mask_shape)
    s = 0
    total_component = mask_shape[0] * mask_shape[1]
    
    n_coeff_kept = int(total_component * min(1, frequence_range[1]))
    n_coeff_to_start = int(total_component * max(0, frequence_range[0]))
    
    while n_coeff_kept > 0:
        for i in range(min(s + 1, mask_shape[0])):
            for j in range(min(s + 1, mask_shape[1])):
                if i + j == s:
                    if n_coeff_to_start > 0:
                        n_coeff_to_start -= 1
                        continue

                    if s % 2:
                        mask[i, j] = 1
                    else:
                        mask[j, i] = 1
                    n_coeff_kept -= 1
                    if n_coeff_kept == 0:
                        return mask
        s += 1
    return mask
