import numpy as np
import torch
import copy
import random
import pickle
import itertools
import json

import eagerpy as ep

from .base import MinimizationAttack
from ..distances import l2
from scipy import fft


def get_module_name(x):
    # splitting is necessary for TensorFlow tensors
    return type(x).__module__.split(".")[0]

def get_raw(x):
    if get_module_name(x).startswith("eagerpy"):
        return x.raw
    else:
        return x

def get_instance(instance):  # type: ignore
    # we use the module name instead of isinstance
    # to avoid importing all the frameworks
    if instance == "torch":  # type: ignore
        return torch.Tensor
    if instance == "numpy":  # type: ignore
        return np.array
    raise ValueError(f"Unknown type: {instance}")

def inner(u, v, axis=1):
    return (u * v).sum(axis)

def norm(u, axis=1):
    e = inner(u, u, axis)
    return e.sqrt()

class Basis:
    def __init__(self, originals, original_labels, random_noise="normal", basis_type="dct", device=-1, *args, **kwargs):
        """
        Args:
            original_labels ([type]): [description]
            random_noise (str, optional): When basis is created, a noise will be added.This noise can be normal or uniform. Defaults to "normal".
            basis_type (str, optional): Type of the basis: DCT, Random, Genetic,. Defaults to "random".
è            device (int, optional): [description]. Defaults to -1.
            args, kwargs: In args and kwargs, there is the basis params:
                    * Random: No parameters                    
                    * DCT:
                            * originals
                            * function: tanh / constant / linear
                            * alpha
                            * beta
                            * lambda
                            * min_frequence
                            * max_frequence
                            * min_dct_value
                            * dct_type: 8x8 or full


        """
        self._originals = originals
        self._original_labels = original_labels
        self.basis_type = basis_type

        self._load_params_type(*args, **kwargs)
        self.device = device

        assert random_noise in ["normal", "uniform"]
        self.random_noise = random_noise

    
    def get_vector(self, index, shape, ortho_with=None, bounds=(0, 1)):
        r = getattr(self, "get_vector_" + self.basis_type)(index, shape, bounds)
        r_flat = r.flatten(0)

        if self.to_clip:
            r_flat = self.clip(r_flat, self.originals[index].flatten())

        if ortho_with is not None:
            #print("Orthogonalize with: {}".format(len(ortho_with)))
            ortho_with = ortho_with.flatten(1)
            n_ortho = len(ortho_with)

            r_flat_repeated = ep.concatenate([r_flat.expand_dims(0)] * n_ortho, axis=0)

            gs_coeff = inner(ortho_with, r_flat_repeated, 1)
            proj = gs_coeff.reshape((n_ortho, 1)) * ortho_with
            proj = proj.sum(0)
            r_flat = r_flat - proj
        norm_r = norm(r_flat, 0)
        r_flat = r_flat / norm_r
        r = r_flat.reshape(shape)
        return r

    def clip(self, v, o):
        lower = - o
        upper = 1 + lower

        where_upper = (v > upper)
        where_lower =  (v < lower)
        where_correct = ep.logical_and(where_upper, where_lower).logical_not()

        new_v = v * where_correct.astype(int) + where_upper.astype(int) * upper + where_lower.astype(int) * lower
        return new_v
    
    def get_vector_dct(self, index, shape, bounds):
        r = np.zeros(shape)
        for channel, dct_channel in enumerate(self.dcts[index]):
            probs = np.random.randint(-2, 1, dct_channel.shape) + 1
            r[channel] = dct_channel * probs

        r = torch.Tensor(idct2_8_8(r) + self._beta * (2*np.random.rand(*r.shape) - 1))
        if self.device is not None:
            r = r.cuda(self.device)
        return ep.astensor(r)

    def get_vector_random(self, index, shape, bounds):
        r = ep.zeros(torch.Tensor, shape)
        if self.device is not None:
            r = r.cuda(self.device)
        r = getattr(ep, self.random_noise)(ep.astensor(r), shape, *bounds)
        return ep.astensor(r)

    def _load_params_type(self, **kwargs):
        if not hasattr(self, "get_vector_" + self.basis_type):
            raise ValueError("Basis {} doesn't exist.")

        if self.basis_type == "dct":
            self._beta = kwargs["beta"] if "beta" in kwargs else 0
            frequence_range = kwargs["frequence_range"] if "frequence_range" in kwargs else (0, 1)

            if "dct_type" not in kwargs or kwargs["dct_type"] == "8x8":
                dct_mask = get_zig_zag_mask(frequence_range, (8, 8))
                self.dcts = np.array([dct2_8_8(np.array(get_raw(image).cpu()), dct_mask) for image in self._originals])
            elif kwargs["dct_type"] == "full":
                dct_mask = get_zig_zag_mask(frequence_range, self._originals.shape[2:])
                self.dcts = np.array([dct2_full(np.array(get_raw(image).cpu()), dct_mask) for image in self.originals])
            else:
                raise ValueError("DCT {} doesn't exist.".format(kwargs["dct_type"]))

            min_value_dct = kwargs["min_value_dct"] if "min_value_dct" in kwargs else 0
            self.dcts *= (abs(self.dcts) > min_value_dct).astype(int)
            
            if "function" not in kwargs or kwargs["function"] == "tanh":
                lambd = kwargs["lambda"] if "lambda" in kwargs else 1
                f = lambda x: np.tanh(lambd * x)
                self.dcts = f(self.dcts)
            elif kwargs["function"] == "identity":
                f = lambda x: x
                self.dcts = f(self.dcts)
            elif kwargs["function"] == "constant":
                self.dcts = (abs(self.dcts) > 0).astype(int)
            else:
                raise ValueError("Function given for DCT is incorrect.")



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
    
    print("DCT Mask: Coefficients from {} to {} will be kept.".format(min_coeff, n_coeff))
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