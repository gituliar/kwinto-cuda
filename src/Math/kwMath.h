#pragma once

#include <cmath>
//#include <vector>

#include "Core/kwCore.h"
#include "Core/kwString.h"


namespace kw
{

template<typename f64>
constexpr f64
    nan = std::numeric_limits<f64>::quiet_NaN();


f64 cdfNormal(f64 x);

Error
    solveTridiagonal(const int xDim, const f64* al, const f64* a, const f64* au, const f64* y, f64* x);

}
