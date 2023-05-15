#pragma once

#include "kwAsset.h"
#include "kwMath.h"
#include "Utils/kwVector2d.h"
#include "Utils/kwConfig.h"

#include <cusparse.h>


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


//  1D PDE:
//
//      0 = 𝒟t V + 𝒜 V
//
//      𝒜 = a0 + ax 𝒟x + axx 𝒟xx
//
//      𝒜 = -r + (r - z²/2) 𝒟x + z²/2 𝒟xx
//
//  Usage:
//     - European Option pricing
//
template<typename Real>
class Fd1d_Gpu {
public:
    // All grids are row-major because of cuSparse's tridiagonal solver
    using CpuGrid = kw::Vector2d<Real, kRowMajor | kCpu>;
    using GpuGrid = kw::Vector2d<Real, kRowMajor | kGpu>;

public:
    Real    m_theta;

    size_t  m_tDim;
    size_t  m_xDim;

    dim3    m_block2d;

    // solution <n × xDim>
    CpuGrid m_v;
    // pde coefficients <n × tDim>
    CpuGrid m_a0, m_ax, m_axx;

    // CUDA-specific stuff

    // solution <n × xDim>
    GpuGrid m__v;
    // pde coefficients <n × tDim>
    GpuGrid m__a0, m__ax, m__axx;

    // x-grid nodes <n × xDim>
    GpuGrid m__x;
    // t-grid nodes <n × tDim>
    GpuGrid m__t;
    // max payoff for early exercise adjustment <n × xDim>
    GpuGrid m__pay;

    // Working Memory

    // Memory buffer used by cuSparse::cusparseSgtsv2StridedBatch (to solve tridiogonal systems)
    // at every backward propagation step of the finite-difference algorithm.
    GpuGrid m__bl, m__b, m__bu;
    GpuGrid m__w;


    // cuSparse-specific stuff
    cusparseHandle_t
            m_cusparseHandle;
    void*   m_cusparseBuf; // GPU memory
    size_t  m_cusparseBufSize;

public:
    const size_t&
        tDim() const { return m_tDim; }
    const size_t&
        xDim() const { return m_xDim; }

    Error
        init(const Config& config, size_t tDim, size_t xDim);
    Error
        free();

    Error
        solve(
            const std::vector<Fd1dPde<Real>>& batch,
            const CpuGrid& tGrid,
            const CpuGrid& xGrid,
            const CpuGrid& vGrid);

    Error
        value(
            const size_t ni,
            const Real s,
            const CpuGrid& xGrid,
            Real& v) const;
};


template<typename Real>
Error
    adjustEarlyExercise(
        const dim3 block2d,
        const size_t n,
        const size_t xDim,
        const Real* pay,
        Real* v
    );

template<typename Real>
Error
    fillB(
        const dim3 block2d,
        const size_t n,
        const size_t xDim,
        const size_t ti,
        const Real theta,
        const Real* t,
        const Real* a0,
        const Real* ax,
        const Real* axx,
        const Real* x,
        Real* bl,
        Real* b,
        Real* bu
    );

template<typename Real>
Error
    fillW(
        const dim3 block2d,
        const size_t n,
        const size_t xDim,
        const size_t ti,
        const Real theta,
        const Real* t,
        const Real* x,
        const Real* a0,
        const Real* ax,
        const Real* axx,
        const Real* v,
        Real* w
    );

}