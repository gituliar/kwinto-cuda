#include "cuda.h"

#include <cstdio>

#include "kwArray.h"


namespace kw
{

// Calc B = [1 - θ dt 𝒜]
template<typename Real>
__global__
void
fillB_g(
    const size_t ti,
    Real theta,

    const kw::Array2d<kw::GPU, Real> t,
    const kw::Array2d<kw::GPU, Real> a0,
    const kw::Array2d<kw::GPU, Real> ax,
    const kw::Array2d<kw::GPU, Real> axx,

    const kw::Array2d<kw::GPU, Real> x,
    kw::Array2d<kw::GPU, Real> bl,
    kw::Array2d<kw::GPU, Real> b,
    kw::Array2d<kw::GPU, Real> bu)
{
    // pde index
    size_t ni = blockIdx.x * blockDim.x + threadIdx.x;
    if (ni >= x.cols())
        return;

    // x-record index
    size_t xDim = x.rows();
    size_t xi = blockIdx.y * blockDim.y + threadIdx.y;
    if (xi >= x.rows())
        return;

    // NOTE: Assuiming uniform dt & dx
    Real dt = t(ni, 1) - t(ni, 0);
    Real dx = x(ni, 1) - x(ni, 0);
    Real inv_dx = 1. / dx;
    Real inv_dx2 = inv_dx * inv_dx;

    auto i = x.index(ni, xi);
    auto j = t.index(ni, ti);
    if (xi == 0)
    {
        bl[i] = 0;
        b[i] = 1 - theta * dt * (a0[j] - ax[j] * inv_dx);
        bu[i] = -theta * dt * (ax[j] * inv_dx);
    }
    else if (xi < xDim - 1)
    {
        bl[i] = -theta * dt * (axx[j] * inv_dx2 - 0.5 * ax[j] * inv_dx);
        b[i] = 1 - theta * dt * (a0[j] - 2 * axx[j] * inv_dx2);
        bu[i] = -theta * dt * (axx[j] * inv_dx2 + 0.5 * ax[j] * inv_dx);
    }
    else
    {
        bl[i] = -theta * dt * (-ax[j] * inv_dx);
        b[i] = 1 - theta * dt * (a0[j] + ax[j] * inv_dx);
        bu[i] = 0;
    }
}

template<typename Real>
kw::Error
fillB(
    const size_t nThreads,
    const size_t xThreads,
    const size_t ti,
    const Real theta,
    const kw::Array2d<kw::GPU, Real>& t,
    const kw::Array2d<kw::GPU, Real>& a0,
    const kw::Array2d<kw::GPU, Real>& ax,
    const kw::Array2d<kw::GPU, Real>& axx,
    const kw::Array2d<kw::GPU, Real>& x,
    kw::Array2d<kw::GPU, Real>& bl,
    kw::Array2d<kw::GPU, Real>& b,
    kw::Array2d<kw::GPU, Real>& bu)
{
    auto n = static_cast<uint32_t>(x.cols());
    auto tDim = static_cast<uint32_t>(t.rows());
    auto xDim = static_cast<uint32_t>(x.rows());

    dim3 block2d(nThreads, xThreads);
    dim3 grid2d;
    grid2d.x = (n + block2d.x - 1) / block2d.x;
    grid2d.y = (xDim + block2d.y - 1) / block2d.y;

    fillB_g<<<grid2d, block2d>>>(ti, theta, t, a0, ax, axx, x, bl, b, bu);

    return "";
};

template
kw::Error
fillB(
    const size_t nThreads,
    const size_t xThreads,
    const size_t ti,
    const float theta,
    const kw::Array2d<kw::GPU, float>& t,
    const kw::Array2d<kw::GPU, float>& a0,
    const kw::Array2d<kw::GPU, float>& ax,
    const kw::Array2d<kw::GPU, float>& axx,
    const kw::Array2d<kw::GPU, float>& x,
    kw::Array2d<kw::GPU, float>& bl,
    kw::Array2d<kw::GPU, float>& b,
    kw::Array2d<kw::GPU, float>& bu);

template
kw::Error
fillB(
    const size_t nThreads,
    const size_t xThreads,
    const size_t ti,
    const double theta,
    const kw::Array2d<kw::GPU, double>& t,
    const kw::Array2d<kw::GPU, double>& a0,
    const kw::Array2d<kw::GPU, double>& ax,
    const kw::Array2d<kw::GPU, double>& axx,
    const kw::Array2d<kw::GPU, double>& x,
    kw::Array2d<kw::GPU, double>& bl,
    kw::Array2d<kw::GPU, double>& b,
    kw::Array2d<kw::GPU, double>& bu);



template<typename Real>
__global__
void
fillW_g(
    const int ti,
    const Real theta,

    const kw::Array2d<kw::GPU, Real> t,
    const kw::Array2d<kw::GPU, Real> a0,
    const kw::Array2d<kw::GPU, Real> ax,
    const kw::Array2d<kw::GPU, Real> axx,

    const kw::Array2d<kw::GPU, Real> x,
    const kw::Array2d<kw::GPU, Real> v,
    kw::Array2d<kw::GPU, Real> w)
{
    // pde index
    size_t ni = blockIdx.x * blockDim.x + threadIdx.x;
    if (ni >= x.cols())
        return;

    // x-record index
    size_t xDim = x.rows();
    size_t xi = blockIdx.y * blockDim.y + threadIdx.y;
    if (xi >= xDim)
        return;

    Real dt = (1 - theta) * (t(ni, 1) - t(ni, 0));
    Real inv_dx = 1. / (x(ni, 1) - x(ni, 0));
    Real inv_dx2 = inv_dx * inv_dx;

    auto i = x.index(ni, xi);
    auto j = a0.index(ni, ti);
    if (xi == 0)
    {
        w[i] = (1 + dt * a0[j]) * v[i] + dt * (ax[j] * inv_dx) * (v(ni, xi + 1) - v(ni, xi));
    }
    else if (xi < xDim - 1)
    {
        w[i] = (1 + dt * a0[j]) * v[i] +
            dt * (0.5 * ax[j] * inv_dx) * (v(ni, xi + 1) - v(ni, xi - 1)) +
            dt * (axx[j] * inv_dx2) * (v(ni, xi + 1) - 2 * v[i] + v(ni, xi - 1));
    }
    else
    {
        w[i] = (1 + dt * a0[j]) * v[i] + dt * (ax[j] * inv_dx) * (v(ni, xi) - v(ni, xi - 1));
    }
}



template<typename Real>
__global__
void
adjustEarlyExercise_g(
    const kw::Array2d<kw::GPU, Real> pay,
    kw::Array2d<kw::GPU, Real> v)
{
    // pde index
    size_t ni = blockIdx.x * blockDim.x + threadIdx.x;
    if (ni >= v.cols())
        return;

    // x-record index
    size_t xDim = v.rows();
    size_t xi = blockIdx.y * blockDim.y + threadIdx.y;
    if (xi >= v.rows())
        return;

    auto& v_ = v(ni, xi);
    auto& pay_ = pay(ni, xi);
    if (v_ < pay_)
        v_ = pay_;
}

template<typename Real>
kw::Error
adjustEarlyExercise(
    const size_t nThreads,
    const size_t xThreads,
    const kw::Array2d<kw::GPU, Real>& pay,
    kw::Array2d<kw::GPU, Real>& v)
{
    auto n = static_cast<uint32_t>(v.cols());
    auto xDim = static_cast<uint32_t>(v.rows());

    dim3 block2d(nThreads, xThreads);
    dim3 grid2d;
    grid2d.x = (n + block2d.x - 1) / block2d.x;
    grid2d.y = (xDim + block2d.y - 1) / block2d.y;

    adjustEarlyExercise_g<<<grid2d, block2d>>>(pay, v);

    return "";
}

template
kw::Error
adjustEarlyExercise<float>(
    const size_t nThreads,
    const size_t xThreads,
    const kw::Array2d<kw::GPU, float>& pay,
    kw::Array2d<kw::GPU, float>& v);

template
kw::Error
adjustEarlyExercise<double>(
    const size_t nThreads,
    const size_t xThreads,
    const kw::Array2d<kw::GPU, double>& pay,
    kw::Array2d<kw::GPU, double>& v);


template<typename Real>
kw::Error
fillW(
    const size_t nThreads,
    const size_t xThreads,
    const size_t ti,
    const Real theta,
    const kw::Array2d<kw::GPU, Real>& t,
    const kw::Array2d<kw::GPU, Real>& a0,
    const kw::Array2d<kw::GPU, Real>& ax,
    const kw::Array2d<kw::GPU, Real>& axx,
    const kw::Array2d<kw::GPU, Real>& x,
    const kw::Array2d<kw::GPU, Real>& v,
    kw::Array2d<kw::GPU, Real>& w)
{
    auto n = static_cast<uint32_t>(x.cols());
    auto tDim = static_cast<uint32_t>(t.rows());
    auto xDim = static_cast<uint32_t>(x.rows());

    dim3 block2d(nThreads, xThreads);
    dim3 grid2d;
    grid2d.x = (n + block2d.x - 1) / block2d.x;
    grid2d.y = (xDim + block2d.y - 1) / block2d.y;

    fillW_g<<<grid2d, block2d>>>(ti, theta, t, a0, ax, axx, x, v, w);

    return "";
}

template
kw::Error
fillW<float>(
    const size_t nThreads,
    const size_t xThreads,
    const size_t ti,
    const float theta,
    const kw::Array2d<kw::GPU, float>& t,
    const kw::Array2d<kw::GPU, float>& a0,
    const kw::Array2d<kw::GPU, float>& ax,
    const kw::Array2d<kw::GPU, float>& axx,
    const kw::Array2d<kw::GPU, float>& x,
    const kw::Array2d<kw::GPU, float>& v,
    kw::Array2d<kw::GPU, float>& w);

template
kw::Error
fillW<double>(
    const size_t nThreads,
    const size_t xThreads,
    const size_t ti,
    const double theta,
    const kw::Array2d<kw::GPU, double>& t,
    const kw::Array2d<kw::GPU, double>& a0,
    const kw::Array2d<kw::GPU, double>& ax,
    const kw::Array2d<kw::GPU, double>& axx,
    const kw::Array2d<kw::GPU, double>& x,
    const kw::Array2d<kw::GPU, double>& v,
    kw::Array2d<kw::GPU, double>& w);

}
