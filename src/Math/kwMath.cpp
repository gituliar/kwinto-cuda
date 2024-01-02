#include <numbers>
#include <vector>

#include "Core/kwTypes.h"
#include "Math/kwMath.h"

using namespace kw;


f64
kw::cdfNormal(f64 x)
{
    return 0.5 * (1 + std::erf(x / std::numbers::sqrt2));
};


Error
kw::solveTridiagonal(const int xDim, const f64* al, const f64* a, const f64* au, const f64* y, f64* x)
{
    if (a[0] == 0)
        return "kw::solveTridiagonal: Error 1";

    if (xDim <= 2)
        // Handle n = 1,2 cases
        return "kw::solveTridiagonal: Error 3";

    std::vector<f64> gam(xDim);

    f64 bet;
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