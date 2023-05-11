#pragma once

#include "kwAsset.h"
#include "kwMath.h"
#include "Utils/kwArray.h"
#include "Utils/kwConfig.h"

#include <cusparse.h>

#include <functional>

namespace kw
{

template<typename Real>
struct Fd1dPde {
    Real    t;

    Real    a0;
    Real    ax;
    Real    axx;
};


//  Finite-difference solver for 1D PDE using the theta scheme
//
//      [1 - θ dt 𝒜] V(t) = [1 + (1 - θ) dt 𝒜] V(t+dt)
//
template<typename Real>
class Fd1d {
public:
    using CpuGrid = kw::Array2d<CPU, Real, kStorage::RowMajor>;

//    struct Pde
//    {
        //Real t;

        //Real a0;
        //Real ax;
        //Real axx;
//    };

private:
    Real    m_theta;

    // x-grid nodes <N × xDim>
    // t-grid nodes <N × tDim>
    // pde coefficients
    CpuGrid m_a0;
    CpuGrid m_ax;
    CpuGrid m_axx;

    // Working Memory
    CpuGrid m_bl;
    CpuGrid m_b;
    CpuGrid m_bu;

    CpuGrid m_w;
    CpuGrid m_v;

public:
    Error   init(size_t bCap, size_t tDim, size_t xDim);
    Error   free();

    Error   solve(
                const std::vector<Fd1dPde<Real>>& batch,
                const CpuGrid& tGrid,
                const CpuGrid& xGrid,
                const CpuGrid& vGrid);
    Error   value(
                const size_t i,
                const Real s,
                const CpuGrid& xGrid,
                Real& v) const;

private:
    Error   solveOne(
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
    using CpuGrid = kw::Array2d<CPU, Real, kStorage::RowMajor>;
    using GpuGrid = kw::Array2d<GPU, Real, kStorage::RowMajor>;
public:
    Real    m_theta;

    size_t  m_nThreads;
    size_t  m_xThreads;

    // solution <N × xDim>
    CpuGrid m_v;
    GpuGrid m__v;

    GpuGrid m__w;

    // x-grid nodes <N × xDim>
    GpuGrid m__x;
    // t-grid nodes <N × tDim>
    GpuGrid m__t;

    // pde coefficients
    CpuGrid m_a0, m_ax, m_axx;
    GpuGrid m__a0, m__ax, m__axx;

    // max payoff for early exercise adjustment
    GpuGrid m__pay;

    // Working Memory

    // Memory buffer used by cuSparse::cusparseSgtsv2StridedBatch (to solve tridiogonal systems)
    // at every backward propagation step of the finite-difference algorithm.
    GpuGrid m__bl, m__b, m__bu;


    cusparseHandle_t
            m_cusparseHandle;
    void*   m_cusparseBuf; // GPU memory
    size_t  m_cusparseBufSize;

public:
    Error   init(const Config& config, size_t tDim, size_t xDim);
    Error   free();

    Error   solve(
                const std::vector<Fd1dPde<Real>>& batch,
                const CpuGrid& tGrid,
                const CpuGrid& xGrid,
                const CpuGrid& vGrid);

    Error   value(
                const size_t ni,
                const Real s,
                const CpuGrid& xGrid,
                Real& v) const;
};


template<typename Real>
Error
adjustEarlyExercise(
    const size_t nThreads,
    const size_t xThreads,
    const Array2d<GPU, Real>& pay,
    Array2d<GPU, Real>& v
);

template<typename Real>
Error
fillB(
    const size_t nThreads,
    const size_t xThreads,
    const size_t ti,
    const Real theta,
    const Array2d<GPU, Real>& t,
    const Array2d<GPU, Real>& a0,
    const Array2d<GPU, Real>& ax,
    const Array2d<GPU, Real>& axx,
    const Array2d<GPU, Real>& x,
    Array2d<GPU, Real>& bl,
    Array2d<GPU, Real>& b,
    Array2d<GPU, Real>& bu
);

template<typename Real>
Error
fillW(
    const size_t nThreads,
    const size_t xThreads,
    const size_t ti,
    const Real theta,
    const Array2d<GPU, Real>& t,
    const Array2d<GPU, Real>& a0,
    const Array2d<GPU, Real>& ax,
    const Array2d<GPU, Real>& axx,
    const Array2d<GPU, Real>& x,
    const Array2d<GPU, Real>& v,
    Array2d<GPU, Real>& w
);

}