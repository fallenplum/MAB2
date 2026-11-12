"""CPU-only continuous sampling relaxations. No inference or external data I/O."""
import math
import numpy as np
from scipy.optimize import minimize_scalar


def direct_srs(n_population, sample_variance, suffix_mean_cost, budget):
    n = min(n_population, budget / suffix_mean_cost)
    variance = sample_variance * (1.0 / n - 1.0 / n_population) if n > 0 else math.inf
    return {"n": n, "variance": max(0.0, variance), "expected_suffix_cost": n * suffix_mean_cost}


def census_probe_neyman(N, counts, variances, aggregate_means, receiver_total, budget):
    """Exact convex optimum of continuous stratified SRS relaxation."""
    counts = np.asarray(counts, float)
    variances = np.asarray(variances, float)
    costs = np.asarray(aggregate_means, float)
    a = (counts / N) ** 2 * variances
    remaining = budget - receiver_total
    if remaining <= 0:
        return {"feasible": False, "variance": None, "n_h": None}
    needed = float(np.dot(counts, costs))
    if remaining >= needed * (1 - 1e-13):
        n = counts.copy()
    else:
        lo, hi = 0.0, 1.0
        def alloc(lam):
            return np.minimum(counts, np.sqrt(a / (lam * costs)))
        while float(np.dot(alloc(hi), costs)) > remaining:
            hi *= 2
        for _ in range(160):
            mid = (lo + hi) / 2
            if float(np.dot(alloc(mid), costs)) > remaining:
                lo = mid
            else:
                hi = mid
        n = alloc(hi)
    variance = float(np.sum(a * (1 / n - 1 / counts)))
    used = receiver_total + float(np.dot(n, costs))
    return {"feasible": True, "variance": max(0., variance), "n_h": n.tolist(), "expected_suffix_cost": used}


def staged_poisson(N, y2_sums, aggregate_totals, receiver_total, budget, base_cost_fn=None):
    """Two live strata: q=max(r1,r2), r_h=q*p_h. Restricted oracle calculation.

    For each maximal stratum, reduce to a scalar ratio t=min(r)/max(r).
    Strictly an expected-budget independent Bernoulli design, not a hard cap.
    Linear-cost branches are one-dimensional convex allocation problems.
    Nonlinear lazy-base branches use grid/local numerical verification, not
    a claimed global certificate unless a separate derivative check is supplied.
    """
    a = np.asarray(y2_sums, float) / (N * N)
    A = np.asarray(aggregate_totals, float)
    candidates = []
    for top in (0, 1):
        low = 1 - top
        def evaluate(t):
            coeff = receiver_total + A[top] + t * A[low]
            if base_cost_fn is None:
                q = min(1., budget / coeff)
            else:
                lo, hi = 0., 1.
                for _ in range(70):
                    mid = (lo+hi)/2
                    if base_cost_fn(mid)+mid*coeff > budget:
                        hi=mid
                    else:
                        lo=mid
                q=lo
            r = np.ones(2) * q
            r[low] *= t
            v = float(np.sum(a * (1 / r - 1)))
            return v, q, r, q * coeff + (0. if base_cost_fn is None else base_cost_fn(q))
        opt = minimize_scalar(lambda t: evaluate(t)[0], bounds=(1e-10, 1.), method="bounded", options={"xatol": 1e-14})
        ts = [float(opt.x), 1.]
        if base_cost_fn is not None:
            grid=np.linspace(1e-6,1.,129)
            val=np.array([evaluate(t)[0] for t in grid])
            ts.append(float(grid[np.argmin(val)]))
            for j in range(1,len(grid)-1):
                if val[j]<=val[j-1] and val[j]<=val[j+1]:
                    local=minimize_scalar(lambda t:evaluate(t)[0],bounds=(grid[j-1],grid[j+1]),method='bounded',options={'xatol':1e-14})
                    ts.append(float(local.x))
        # Include the q=1 kink explicitly.
        kink = (budget - receiver_total - A[top] - (0. if base_cost_fn is None else base_cost_fn(1.))) / A[low]
        if 0 < kink < 1:
            ts.append(float(kink))
        for t in ts:
            v, q, r, used = evaluate(t)
            candidates.append({"variance": max(0., v), "q": q, "p_h": (r / q).tolist(), "inclusion_h": r.tolist(), "expected_suffix_cost": used})
    return min(candidates, key=lambda d: d["variance"])


def direct_poisson(N, y2_sum, suffix_total, budget):
    q = min(1., budget / suffix_total)
    return {"q": q, "variance": y2_sum / N**2 * (1 / q - 1), "expected_suffix_cost": q * suffix_total}


def lazy_srs_base_cost(group_sizes, group_costs, N, n):
    """Continuous extension of exact group-touch expectation; exact at integer n.

    Probability no item in a group is selected under SRS is C(N-m,n)/C(N,n).
    The gamma extension for noninteger n is a relaxation, not a random design.
    """
    from scipy.special import gammaln
    m = np.asarray(group_sizes, float)
    costs = np.asarray(group_costs, float)
    logp = np.full(len(m), -np.inf)
    mask = n <= N - m
    logp[mask] = gammaln(N-m[mask]+1)-gammaln(N-m[mask]-n+1)+gammaln(N-n+1)-gammaln(N+1)
    touch = -np.expm1(logp)
    return float(np.dot(costs, touch))


def self_check():
    """Independent numerical checks of allocation and finite-population formula."""
    from scipy.optimize import minimize
    from itertools import combinations
    y = np.array([0.,0.,-1.,1.,0.,1.,-1.,0.])
    N = len(y)
    exact = []
    for idx in combinations(range(N), 3):
        exact.append(float(y[list(idx)].mean()))
    predicted = y.var(ddof=1) * (1/3-1/N)
    assert abs(np.var(exact)-predicted) < 1e-12
    counts, variances, costs = np.array([35.,45.]), np.array([.02,.15]), np.array([2.,3.])
    a = (counts/100)**2*variances
    oracle = census_probe_neyman(100,counts,variances,costs,100,180)
    opt = minimize(lambda n: float(np.sum(a*(1/n-1/counts))), [10.,20.], bounds=[(1e-8,35.),(1e-8,45.)], constraints=[{"type":"ineq","fun":lambda n: 80-float(n@costs)}], method="SLSQP", options={"ftol":1e-13,"maxiter":2000})
    assert opt.success and abs(oracle['variance']-opt.fun)<1e-10
    st = staged_poisson(100,[1.,7.],[80.,100.],60,80)
    aa = np.array([1.,7.])/100**2
    def loss(x):
        return float(np.sum(aa*(1/x[1:]-1)))
    opt2 = minimize(loss,[.4,.15,.4],method="SLSQP",bounds=[(1e-8,1)]*3,constraints=[{"type":"ineq","fun":lambda x: 80-60*x[0]-80*x[1]-100*x[2]},{"type":"ineq","fun":lambda x:x[0]-x[1]},{"type":"ineq","fun":lambda x:x[0]-x[2]}],options={"ftol":1e-13,"maxiter":2000})
    assert opt2.success and abs(st['variance']-opt2.fun)<1e-10
    assert census_probe_neyman(100,counts,variances,costs,100,100+float(counts@costs))["variance"]==0
    assert abs(lazy_srs_base_cost([4,4],[10,20],8,1)-15)<1e-10
    return {"enumerated_SRS_subsets":len(exact),"SRS_variance_error":abs(np.var(exact)-predicted),"Neyman_vs_SLSQP_error":abs(oracle['variance']-opt.fun),"Poisson_vs_SLSQP_error":abs(st['variance']-opt2.fun),"status":"PASS"}


if __name__ == "__main__":
    import json
    print(json.dumps(self_check(), indent=2))
