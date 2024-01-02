#pragma once

#include <vector>

#include "Core/kwAsset.h"
#include "Core/kwConfig.h"
#include "Core/kwGrid2d.h"


namespace kw
{

// Container for a 1D PDE.
//
// It should be possible to make coefficients time-dependent (but kept cont for now).
//
struct Fd1dPde {
    f64 t;

    f64 a0;
    f64 ax;
    f64 axx;

    bool earlyExercise;
};

//  1D PDE:
//
//      0 = 𝒟t V + 𝒜 V
//
//      𝒜 = a0 + ax 𝒟x + axx 𝒟xx
//
//      𝒜 = -r + (r - z²/2) 𝒟x + z²/2 𝒟xx
//
//  Finite-difference solver for a 1D PDE based on the theta scheme:
//
//      [1 - θ dt 𝒜] V(t) = [1 + (1 - θ) dt 𝒜] V(t+dt)
//
class Fd1d {
public:
    using CpuGrid = kw::Grid2d;

private:
    f64 m_theta;

    u64 m_tDim;
    u64 m_xDim;

    // pde coefficients
    // t-grid <n × tDim>
    CpuGrid m_a0;
    CpuGrid m_ax;
    CpuGrid m_axx;

    // x-grid <n × xDim>
    CpuGrid m_bl;
    CpuGrid m_b;
    CpuGrid m_bu;

    CpuGrid m_w;
    CpuGrid m_v;

public:
    const u64&
        tDim() const { return m_tDim; }
    const u64&
        xDim() const { return m_xDim; }

    Error
        init(u64 tDim, u64 xDim);

    Error
        solve(
            const std::vector<Fd1dPde>& batch,
            const CpuGrid& tGrid,
            const CpuGrid& xGrid,
            const CpuGrid& vGrid);
    Error
        value(
            const u64 i,
            const f64 s,
            const CpuGrid& xGrid,
            f64& v) const;

private:
    Error
        solveOne(
            const u64 ni,
            const bool earlyExercise,
            const CpuGrid& tGrid,
            const CpuGrid& xGrid,
            const CpuGrid& vGrid);
};


}