"""(1+1)-EA (section 5): start from a random mask (bits drawn with probability 0.5); standard bit
mutation flips each bit independently with rate 1/K (an offspring may equal its parent and is still
evaluated); the offspring is accepted if its fitness is greater than or equal."""

from ._base import budgeted, random_mask


@budgeted
def run(evaluator, K, rng, trace):
    current = random_mask(K, rng)
    f_cur = evaluator(current)
    while True:
        flips = rng.random(K) < 1.0 / K
        cand = current ^ flips
        f_new = evaluator(cand)
        if f_new >= f_cur:
            current, f_cur = cand, f_new
