#pragma once

#include <functional>

#include "cusparse.h"

#include "kwArray.h"
#include "kwAsset.h"
#include "kwMath.h"

namespace kw
{

template<typename Real>
struct Fd1dPde
{
    Real
        k;
    // const (for now)
    Real
        q;
    Real
        r;
    Real
        s;
    Real
        t;
    Real
        z;

    bool
        earlyExercise;

    std::function<void(const Real r, const Real q, const Real z, Real& a0, Real& ax, Real& axx)>
        abc;
    std::function<void(const Real s, const Real k, Real& pay)>
        payoff;
};


struct Fd1dConfig
{
    size_t
        pdeCount;
    double
        theta;
    size_t
        tDim;
    size_t
        xDim;
    size_t
        nThreads;
    size_t
        xThreads;

    Fd1dConfig() :
        pdeCount{ 0 },
        theta{ 0.5 },
        tDim{ 101 },
        xDim{ 101 },
        nThreads{ 1 },
        xThreads{ 64 }
    {};
};


template<typename Real>
Error
Fd1dPdeFor(
    const std::vector<Option>& assets,
    const Fd1dConfig& config,
    std::vector<Fd1dPde<Real>>& pdes)
{
    pdes.clear();
    pdes.reserve(assets.size());

    for (const auto& asset : assets)
    {
        auto& pde = pdes.emplace_back();

        pde.k = asset.k;
        pde.q = asset.q;
        pde.r = asset.r;
        pde.s = asset.s;
        pde.t = asset.t;
        pde.z = asset.z;

        pde.earlyExercise = asset.e;

        if (asset.w == kParity::Put)
            pde.payoff = [](const Real x, const Real k, Real& pay)
            {
                pay = std::max<Real>(0, k - std::exp(x));
            };
        else
            pde.payoff = [](const Real x, const Real k, Real& pay)
            {
                pay = std::max<Real>(0, std::exp(x) - k);
            };

        pde.abc = [](const Real r, const Real q, const Real z, Real& a0, Real& ax, Real& axx)
        {
            a0 = -r;
            ax = r - q - z * z / 2;
            axx = z * z / 2;
        };
    }

    return "";
};


//  Finite-difference solver for 1D PDE using the theta scheme
//
//      [1 - θ dt 𝒜] V(t) = [1 + (1 - θ) dt 𝒜] V(t+dt)
//
template<typename Real>
class Fd1d
{
    using Array2d = kw::Array2d<CPU, Real, kStorage::ColMajor>;

private:
    Real
        m_theta;

    // total PDEs count
    size_t
        m_n;
    size_t
        m_cap;

    // x-grid nodes <N × xDim>
    Array2d
        m_x;
    size_t
        m_xDim;
    Array2d
        m_v;

    // t-grid nodes <N × tDim>
    Array2d
        m_t;
    size_t
        m_tDim;
    // pde coefficients
    Array2d
        m_a0;
    Array2d
        m_ax;
    Array2d
        m_axx;

    // Working Memory
    Array2d
        m_bl;
    Array2d
        m_b;
    Array2d
        m_bu;

    Array2d
        m_w;

public:
    Error
        allocate(const Fd1dConfig& config);
    Error
        free();

    Error
        solve(const std::vector<Fd1dPde<Real>>& pdes);

    Error
        value(const size_t i, const Real s, Real& v) const;

private:
    Error
        solve(const std::vector<Fd1dPde<Real>>& pdes, uint32_t ni);
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
class Fd1d_Gpu
{
public:
    Real
        m_theta;

    // total PDEs count
    size_t
        m_n;
    size_t
        m_cap;

    size_t
        m_nThreads;
    size_t
        m_xThreads;

    // final solution
    kw::Array2d<CPU, Real>
        m__v, m__x;
    kw::Array2d<CPU, Real>
        m__t, m__a0, m__ax, m__axx;

    // x-grid nodes <N × xDim>
    kw::Array2d<GPU, Real>
        m_x;
    size_t
        m_xDim;
    kw::Array2d<GPU, Real>
        m_v;

    // t-grid nodes <N × tDim>
    kw::Array2d<GPU, Real>
        m_t;
    size_t
        m_tDim;
    // pde coefficients
    kw::Array2d<GPU, Real>
        m_a0;
    kw::Array2d<GPU, Real>
        m_ax;
    kw::Array2d<GPU, Real>
        m_axx;
    kw::Array2d<CPU, Real>
        m__pay;

    // max payoff for early exercise adjustment
    kw::Array2d<GPU, Real>
        m_pay;

    // Working Memory

    // Memory buffer used by cuSparse::cusparseSgtsv2StridedBatch (to solve tridiogonal systems)
    // at every backward propagation step of the finite-difference algorithm.
    kw::Array2d<GPU, Real>
        m_bl;
    kw::Array2d<GPU, Real>
        m_b;
    kw::Array2d<GPU, Real>
        m_bu;

    kw::Array2d<GPU, Real>
        m_w;

    cusparseHandle_t
        m_cusparseHandle;
    void*
        m_cusparseBuf; // GPU memory
    size_t
        m_cusparseBufSize;

public:
    Error
        allocate(const Fd1dConfig& config);
    Error
        free();

    Error
        solve(const std::vector<Fd1dPde<Real>>& pdes);

    Error
        value(const size_t i, const Real s, Real& v) const;
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