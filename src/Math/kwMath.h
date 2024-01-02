#pragma once

#include "Core/kwTypes.h"


namespace kw
{

f64 cdfNormal(f64 x);

Error
    solveTridiagonal(const int xDim, const f64* al, const f64* a, const f64* au, const f64* y, f64* x);

}
