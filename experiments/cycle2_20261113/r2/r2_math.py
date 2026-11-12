"""Pure prospective R2 mathematics. No file loading, model imports, GPU, or scores."""
from math import sqrt, isfinite

def _nonnegative(*xs):
    if not all(isfinite(x) and x >= 0 for x in xs):
        raise ValueError("Expected finite nonnegative moments or costs")

def _counts(*ns):
    if not all(type(n) is int and n > 0 for n in ns):
        raise ValueError("Sample counts must be positive fixed integers")

def transport_variance(v_old, v_drift, m, n):
    """Only for independent old bank and drift panel with common source law."""
    _nonnegative(v_old, v_drift); _counts(m,n)
    return v_old/m + v_drift/n

def control_variate_variance(v_a, v_b, covariance, m, n, beta):
    _nonnegative(v_a,v_b); _counts(m,n)
    if not isfinite(beta) or not isfinite(covariance):
        raise ValueError("Non-finite coefficient or covariance")
    if covariance*covariance > v_a*v_b + 1e-12:
        raise ValueError("Moments violate covariance positive semidefiniteness")
    result=v_b/n+beta*beta*v_a*(1/n+1/m)-2*beta*covariance/n
    if result < -1e-12: raise ValueError("Negative variance")
    return max(result,0.0)

def population_optimal_beta(v_a, covariance, m, n):
    """Diagnostic population coefficient, never a fitted operational policy."""
    _nonnegative(v_a); _counts(m,n)
    if not isfinite(covariance): raise ValueError("Non-finite covariance")
    if v_a==0:
        if covariance != 0: raise ValueError("Zero source variance requires zero covariance")
        return 0.0
    return covariance/(v_a*(1+n/m))

def independent_bank_necessary_condition(v_a,v_b,m,n_direct):
    _nonnegative(v_a,v_b); _counts(m,n_direct)
    return v_a/m < v_b/n_direct

def from_scratch_coefficient(v_a,v_d,c_a,c_d):
    """Continuous-allocation variance * budget envelope; not an empirical result."""
    _nonnegative(v_a,v_d,c_a,c_d)
    if c_a==0 or c_d==0: raise ValueError("Acquisition costs must be positive")
    return (sqrt(v_a*c_a)+sqrt(v_d*c_d))**2

def safe_risk_ratio(transport_risk,direct_risk,near_zero=1e-4):
    _nonnegative(transport_risk,direct_risk,near_zero)
    if direct_risk <= near_zero:
        return {"status":"INCONCLUSIVE_NEAR_ZERO_DENOMINATOR","ratio":None}
    return {"status":"DEFINED","ratio":transport_risk/direct_risk}

def template_cost(k):
    if type(k) is not int or not 0<=k<=6: raise ValueError("Unsupported editable count")
    direct=75008+k*16768
    drift=75008+33408+2*k*16768
    return {"direct_calls":9+2*k,"four_world_calls":13+4*k,
            "direct_token_upper":direct,"four_world_token_upper":drift}

def pooled_no_change_variance(v_a,m,n):
    _nonnegative(v_a); _counts(m,n)
    return v_a/(m+n)

