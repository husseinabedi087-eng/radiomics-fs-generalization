"""Bit-flip hill climbing (section 5): start from a random mask (bits drawn with probability 0.5);
each step flips one uniformly chosen bit; the move is accepted if fitness is greater than or equal."""

from ._base import budgeted, random_mask


@budgeted
def run(evaluator, K, rng, trace):
    current = random_mask(K, rng)
    f_cur = evaluator(current)
    while True:
        cand = current.copy()
        j = int(rng.integers(K))
        cand[j] = not cand[j]
        f_new = evaluator(cand)
        if f_new >= f_cur:
            current, f_cur = cand, f_new
