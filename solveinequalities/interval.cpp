#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <algorithm>

namespace py = pybind11;
using Interval = std::pair<double, double>;
using IntervalList = std::vector<Interval>;

constexpr double EPS = 1e-10;
constexpr double INF = std::numeric_limits<double>::infinity();

// solve a single quadratic inequality: ax^2 + bx + c ≤ 0
IntervalList solve_single_quadratic(double a, double b, double c) {
    a = std::abs(a) < EPS ? 0.0 : a;
    b = std::abs(b) < EPS ? 0.0 : b;
    c = std::abs(c) < EPS ? 0.0 : c;

    if (a == 0) {
        // Linear inequality: bx + c <= 0
        if (b == 0) {
            if (c <= 0) return { {-INF, INF} };
            else throw std::runtime_error("No solution: constant > 0");
        }
        double root = -c / b;
        return b > 0 ? IntervalList{ {-INF, root} } : IntervalList{ {root, INF} };
    }

    double delta = b * b - 4 * a * c;

    if (delta < -EPS) {
        return (a < 0) ? IntervalList{ {-INF, INF} } : IntervalList{};
    }

    double sqrt_delta = (delta > 0) ? std::sqrt(delta) : 0.0;
    double x1 = (-b - sqrt_delta) / (2 * a);
    double x2 = (-b + sqrt_delta) / (2 * a);
    if (x1 > x2) std::swap(x1, x2);

    if (a > 0) {
        if (delta < EPS)
            return { {x1, x1} };  // single point
        return { {x1, x2} };     // inside root interval
    } else {
        if (delta < EPS)
            return { {-INF, x1}, {x1, INF} };  // two touching intervals
        return { {-INF, x1}, {x2, INF} };      // union of outer intervals
    }
}

// Intersects two lists of intervals
IntervalList intersect_intervals(const IntervalList& a, const IntervalList& b) {
    IntervalList result;
    size_t i = 0, j = 0;

    while (i < a.size() && j < b.size()) {
        double start = std::max(a[i].first, b[j].first);
        double end = std::min(a[i].second, b[j].second);

        if (start <= end)  // allow equality to preserve single-point intervals
            result.emplace_back(start, end);

        if (a[i].second < b[j].second)
            ++i;
        else
            ++j;
    }

    return result;
}

// solve a system of inequalities: aix2 + bix + ci ≤ 0 for all i
IntervalList solve_system_of_quadratics(const std::vector<double>& a,
                                        const std::vector<double>& b,
                                        const std::vector<double>& c) {
    if (a.size() != b.size() || b.size() != c.size())
        throw std::runtime_error("Input vectors must be the same size.");

    IntervalList result = { {-INF, INF} };

    for (size_t i = 0; i < a.size(); ++i) {
        IntervalList single = solve_single_quadratic(a[i], b[i], c[i]);
        result = intersect_intervals(result, single);
        if (result.empty()) break;  // no solution exists
    }

    return result;
}

std::vector<std::pair<double, double>> interval_intersection(const std::vector<double>& A,
                                                             const std::vector<double>& B) {
    double lower = -std::numeric_limits<double>::infinity();
    double upper =  std::numeric_limits<double>::infinity();

    for (size_t i = 0; i < A.size(); ++i) {
        double a = A[i];
        double b = B[i];

        if (std::isnan(a) || std::isnan(b)) {
            continue;  // Skip invalid constraints
        }

        if (a == 0.0) {
            if (b <= 0.0) {
                throw std::runtime_error("Infeasible constraint: 0 < b where b <= 0");
            }
            continue;  // 0·x < b, always true if b > 0
        }

        if (a > 0) {
            if (a == 0.0) throw std::runtime_error("Division by zero in a > 0 case");
            double bound = b / a;
            if (!std::isfinite(bound)) throw std::runtime_error("Non-finite upper bound");
            upper = std::min(upper, bound);
        } else if (a < 0) {
            if (a == 0.0) throw std::runtime_error("Division by zero in a < 0 case");
            double bound = b / a;
            if (!std::isfinite(bound)) throw std::runtime_error("Non-finite lower bound");
            lower = std::max(lower, bound);
        }
    }

    if (lower < upper) {
        return { {lower, upper} };
    } else {
        return {};  // No feasible region
    }
}

PYBIND11_MODULE(interval, m) {
    m.def("interval_intersection", &interval_intersection,
          "Compute intersection of Ax < B and raise on division by zero");
    m.def("solve_system_of_quadratics", &solve_system_of_quadratics,
    "Solve system of quadratic inequalities ax^2 + bx + c < 0 and return intersected intervals");
}
