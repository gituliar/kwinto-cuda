#pragma once

#include <cmath>
#include <numbers>
#include <vector>

#include "kwArray.h"


namespace kw
{

template<typename Real>
constexpr Real
    nan = std::numeric_limits<Real>::quiet_NaN();


template<typename Real>
Real
    cdfNormal(Real x)
{
    return 0.5 * (1 + std::erf(x / std::numbers::sqrt2));
};


template<typename Real>
Error
    solveTridiagonal(const int xDim, const Real* al, const Real* a, const Real* au, const Real* y, Real* x)
{
    if (a[0] == 0)
        return "kw::solveTridiagonal: Error 1";

    if (xDim <= 2)
        // Handle n = 1,2 cases
        return "kw::solveTridiagonal: Error 3";

    std::vector<Real> gam(xDim);

    Real bet;
    x[0] = y[0] / (bet = a[0]);
    for (auto j = 1; j < xDim; j++)
    {
        gam[j] = au[j - 1] / bet;
        bet = a[j] - al[j] * gam[j];

        if (bet == 0)
            return "kw::solveTridiagonal::SolveTridiagonal: Error 2";

        x[j] = (y[j] - al[j] * x[j - 1]) / bet;
        if (x[j] < 0)
            continue;
    };

    for (auto j = xDim - 2; j >= 0; --j)
    {
        x[j] -= gam[j + 1] * x[j + 1];
    }

    return "";
};

}
