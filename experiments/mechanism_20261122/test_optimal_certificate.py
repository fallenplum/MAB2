"""Independent integer completion oracle for the finite-kernel audit."""
import itertools
import unittest

import numpy as np

from optimal_certificate import optimal


def possible_row_scores(row, mask, target, source, base, condition, labels):
    # Distributions and source probabilities have denominator 8. Enumerate
    # each possible row and every binary terminal completion for this target.
    point_masses = [tuple(8 * int(i == j) for j in range(4)) for i in range(4)]
    distributions = [condition[row]] if mask[0] else point_masses
    possibilities = [[int(y == target)] if observed else [0, 1]
                     for observed, y in zip(mask[1:], labels[row])]
    values = [source[row] * sum((p - b) * hit for p, b, hit in zip(dist, base, hits))
              for dist in distributions for hits in itertools.product(*possibilities)]
    return min(values), max(values)


def brute(source, base, condition, labels, prices, targets):
    # The other two source states have exact mass zero. Enumerate all
    # 2^10 transcripts for the active states; costs and tests stay integral.
    masks = list(itertools.product((0, 1), repeat=5))
    options = []
    for row in (0, 1):
        options.append([(sum(a*b for a,b in zip(mask, prices[row])),
                         [possible_row_scores(row, mask, y, source, base, condition, labels)
                          for y in targets]) for mask in masks])
    cutoff = 3.5  # scores have denominator 64: half-step avoids float ties
    return min(ca + cb for ca, a in options[0] for cb, b in options[1]
               if all(l + v > cutoff or u + w < cutoff for (l,u),(v,w) in zip(a,b)))


class TestOptimal(unittest.TestCase):
    def test_matches_enumerated_integer_completions(self):
        for seed in range(12):
            rng = np.random.default_rng(seed)
            source = np.array([3,5,0,0])
            base = rng.multinomial(8, [.25]*4)
            condition = rng.multinomial(8, [.25]*4, size=4)
            labels = rng.choice(['x','y','z','INVALID'],size=(4,4))
            prices = rng.integers(1,8,(4,5))
            for targets in (['x'], ['new'], ['x','y','z','new']):
                cost = optimal(source/8,base/8,condition/8,labels,prices[:,0],
                               prices[:,1:],targets,prices.sum(),tau=3.5/64)
                self.assertEqual(cost,brute(source,base,condition,labels,prices,targets))


if __name__ == '__main__':
    unittest.main()
