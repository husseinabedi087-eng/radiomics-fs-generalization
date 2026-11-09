"""Transfer functions and binarization rules (PREREG_STAGE1 section 5).

S-shaped (sigmoid), Kennedy & Eberhart (1997), "A discrete binary version of the particle swarm
algorithm", Proc. IEEE SMC, 4104-4108:
    bit_d = 1  if  u_d < 1 / (1 + exp(-x_d))   else 0,        u_d ~ U(0, 1)

V-shaped (|tanh|), Mirjalili & Lewis (2013), "S-shaped versus V-shaped transfer functions for
binary particle swarm optimization", Swarm and Evolutionary Computation 9, 1-14:
    bit_d = 1 - prev_d  if  u_d < |tanh(x_d)|   else prev_d,  u_d ~ U(0, 1)

x is the velocity (binary PSO) or the continuous position (DE, GWO, WOA); prev is the agent's
current mask. Both rules are symmetric under bit inversion combined with x -> -x.
"""

import numpy as np

TRANSFERS = ("S", "V")


def s_shaped(x):
    return 1.0 / (1.0 + np.exp(-np.asarray(x, dtype=float)))


def v_shaped(x):
    return np.abs(np.tanh(np.asarray(x, dtype=float)))


def binarize(x, prev_mask, transfer, rng):
    """New mask from continuous values x. prev_mask is required for the V-shaped rule."""
    x = np.asarray(x, dtype=float)
    u = rng.random(x.shape)
    if transfer == "S":
        return u < s_shaped(x)
    if transfer == "V":
        if prev_mask is None:
            raise ValueError("the V-shaped rule needs the previous mask")
        return np.logical_xor(np.asarray(prev_mask, dtype=bool), u < v_shaped(x))
    raise ValueError(f"transfer must be one of {TRANSFERS}")
