import numpy as np
from mpmath import mp
mp.dps = 500

def construct_p_q_t(a, b, A = None):
    if A is None:
        p = np.dot(b.T, b)
        q = 2 * np.dot(b.T, a)
        t = np.dot(a.T, a)
        return p, q, t
    
    p = np.dot(b.T, np.dot(A, b))
    q = 2 * np.dot(b.T, np.dot(A, a))
    t = np.dot(a.T, np.dot(A, a))
    return p, q, t

def compute_p_value(intervals, test_stat, etaT_Sigma_eta):
    denominator = 0
    numerator = None

    for i in intervals:
        leftside, rightside = i
        if leftside <= test_stat <= rightside:
            numerator = denominator + mp.ncdf(test_stat / np.sqrt(etaT_Sigma_eta)) - mp.ncdf(leftside / np.sqrt(etaT_Sigma_eta))
        denominator += mp.ncdf(rightside / np.sqrt(etaT_Sigma_eta)) - mp.ncdf(leftside / np.sqrt(etaT_Sigma_eta))
    cdf = float(numerator / denominator)
    return 2 * min(cdf, 1 - cdf)

def interval_intersection(a, b):
    i = j = 0
    result = []
    while i < len(a) and j < len(b):
        a_start, a_end = a[i]
        b_start, b_end = b[j]

        # Calculate the potential intersection
        start = max(a_start, b_start)
        end = min(a_end, b_end)

        # If the interval is valid, add to results
        if start < end:
            result.append((start, end))

        # Move the pointer which ends first
        if a_end < b_end:
            i += 1
        else:
            j += 1
    return result


def interval_union(a, b):
    # Merge the two sorted interval lists into one sorted list
    merged = []
    i = j = 0
    while i < len(a) and j < len(b):
        if a[i][0] < b[j][0]:
            merged.append(a[i])
            i += 1
        else:
            merged.append(b[j])
            j += 1
    # Add any remaining intervals from a or b
    merged.extend(a[i:])
    merged.extend(b[j:])

    # Merge overlapping intervals
    if not merged:
        return []

    result = [merged[0]]
    for current in merged[1:]:
        last = result[-1]
        if current[0] < last[1]:
            # Overlapping or adjacent, merge them
            new_start = last[0]
            new_end = max(last[1], current[1])
            result[-1] = (new_start, new_end)
        else:
            result.append(current)
    return result

def solve_quadratic_inequality(a, b, c):
    """ ax^2 + bx +c <= 0 """
    a, b, c = float(a), float(b), float(c)
    if abs(a) < 1e-8:
        a = 0
    if abs(b) < 1e-8:
        b = 0
    if abs(c) < 1e-8:
        c = 0
    if a == 0:
        if b > 0:
            return [(-np.inf, -c / b)]
        elif b == 0:
            if c <= 0:
                return [(-np.inf, np.inf)]
            else:
                print('Error bx + c')
                return 
        else:
            return [(-c / b, np.inf)]

    delta = b*b - 4*a*c
    if delta < 0:
        if a < 0:
            return [(-np.inf, np.inf)]
        else:
            print("Error to find interval. ")

    x1 = (- b - np.sqrt(delta)) / (2.0*a)
    x2 = (- b + np.sqrt(delta)) / (2.0*a)

    # x1 = np.around(x1, 8)
    # x2 = np.around(x2, 8)
    if a < 0:
        return [(-np.inf, x2),(x1, np.inf)]
    return [(x1,x2)]


# numba_version.py
from numba import njit
import math
from math import inf

@njit
def _solve_quadratic_core(a, b, c):
    """
    Numba-jittable core that returns a small result code and up to two endpoints.
    Codes:
      0 -> all real numbers (return (-inf, +inf))
      1 -> empty set (no solution)
      2 -> single closed interval [v1, v2]
      3 -> (-inf, v1]   (linear case b > 0)
      4 -> [v1, +inf)   (linear case b < 0)
      5 -> two intervals (-inf, v1] U [v2, +inf)  (a < 0, delta >= 0)
    """
    tol = 1e-10
    if abs(a) < tol:
        a = 0.0
    if abs(b) < tol:
        b = 0.0
    if abs(c) < tol:
        c = 0.0

    # Linear or constant cases
    if a == 0.0:
        if b == 0.0:
            if c <= 0.0:
                return (0, 0.0, 0.0)  # all reals
            else:
                return (1, 0.0, 0.0)  # no solution
        else:
            # bx + c <= 0  ->  x <= -c/b if b>0, else x >= -c/b
            val = -c / b
            if b > 0.0:
                return (3, val, 0.0)
            else:
                return (4, val, 0.0)

    # Quadratic case
    delta = b * b - 4.0 * a * c
    if delta < 0.0:
        if a < 0.0:
            return (0, 0.0, 0.0)  # always <= 0
        else:
            return (1, 0.0, 0.0)  # always > 0 -> no solution

    sqrt_delta = math.sqrt(delta)
    r1 = (-b - sqrt_delta) / (2.0 * a)
    r2 = (-b + sqrt_delta) / (2.0 * a)
    left = r1 if r1 <= r2 else r2
    right = r2 if r2 >= r1 else r1

    if a > 0.0:
        # Upward parabola -> between roots satisfies <= 0
        return (2, left, right)
    else:
        # Downward parabola -> outside roots satisfies <= 0
        return (5, left, right)


def solve_quadratic_inequality_numba(a, b, c, round_digits=8):
    """
    Wrapper that calls the numba core and returns list-of-interval tuples.
    Each tuple is (start, end) where start or end may be math.inf or -math.inf.
    Endpoints are rounded to `round_digits`.
    """
    code, v1, v2 = _solve_quadratic_core(float(a), float(b), float(c))
    # small local rounding helper
    def r(x):
        # keep infinities as-is
        if math.isinf(x):
            return x
        return round(x, round_digits)

    if code == 0:
        return [(-inf, inf)]
    elif code == 1:
        return []  # empty solution
    elif code == 2:
        return [(r(v1), r(v2))]
    elif code == 3:
        return [(-inf, r(v1))]
    elif code == 4:
        return [(r(v1), inf)]
    elif code == 5:
        return [(-inf, r(v1)), (r(v2), inf)]
    else:
        # should not happen
        print("Error to find interval. ")
        return []
