#pragma once

#include "Core/kwAsset.h"
#include "Core/kwConfig.h"
#include "Core/kwGrid2d.h"


namespace kw
{

/// Partial-Differential Equation with _constant_ coefficients.
///
///     𝒟t V + 𝒜 V = 0,    where    𝒜 = a0 + ax 𝒟x + axx 𝒟xx
///
struct Fd1dPde {
    f64 t;

    f64 a0;
    f64 ax;
    f64 axx;

    bool earlyExercise;
};

/// Finite-Difference solver for a 1D PDE based on the Crank-Nicolson scheme:
///
///     [1 - θ dt 𝒜] V(t) = [1 + (1 - θ) dt 𝒜] V(t+dt)
///
class Fd1d {
private:
    f64 m_theta = 0.5;

    /// Equations to solve <n × tDim>
    ///
    Grid2d m_a0;
    Grid2d m_ax;
    Grid2d m_axx;

    /// Tridiagonal systems <n × xDim>
    ///
    Grid2d m_bl;
    Grid2d m_b;
    Grid2d m_bu;

    Grid2d m_w;
    Grid2d m_v;

public:
    Error
        solve(const vector<Fd1dPde>& pdes, const Grid2d& t, const Grid2d& x, const Grid2d& v);
    Error
        value(const u64 i, const f64 s, const Grid2d& x, f64& v) const;

private:
    Error
        solveOne(const u64 ni, const bool earlyExercise, const Grid2d& t, const Grid2d& x, const Grid2d& v);
};


}