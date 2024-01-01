#pragma once

#include "kwAsset.h"
#include "kwMath.h"
#include "Utils/kwVector2d.h"
#include "Utils/kwConfig.h"


namespace kw
{

// Container for a 1D PDE.
//
// It should be possible to make coefficients time-dependent (but kept cont for now).
//
template<typename Real>
struct Fd1dPde {
    Real    t;

    Real    a0;
    Real    ax;
    Real    axx;
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
template<typename Real>
class Fd1d {
public:
    using CpuGrid = kw::Vector2d<Real, kColMajor | kCpu>;

private:
    Real    m_theta;

    size_t  m_tDim;
    size_t  m_xDim;

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
    const size_t&
        tDim() const { return m_tDim; }
    const size_t&
        xDim() const { return m_xDim; }

    Error
        init(size_t tDim, size_t xDim);

    Error
        solve(
            const std::vector<Fd1dPde<Real>>& batch,
            const CpuGrid& tGrid,
            const CpuGrid& xGrid,
            const CpuGrid& vGrid);
    Error
        value(
            const size_t i,
            const Real s,
            const CpuGrid& xGrid,
            Real& v) const;

private:
    Error
        solveOne(
            const uint32_t ni,
            const CpuGrid& tGrid,
            const CpuGrid& xGrid,
            const CpuGrid& vGrid);
};


}