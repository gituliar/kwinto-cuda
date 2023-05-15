#include "Utils/kwVector2d.h"

#include "cuda.h"

#include <cstdio>


namespace kw
{

template<typename Real>
__global__
void
adjustEarlyExercise_cuda(
    const size_t n,
    const size_t xDim,
    const Real* pay,
    Real* v)
{
    // pde index
    size_t ni = blockIdx.x * blockDim.x + threadIdx.x;
    if (ni >= n)
        return;

    // x-record index
    //size_t xDim = pay.rows();
    size_t xi = blockIdx.y * blockDim.y + threadIdx.y;
    if (xi >= xDim)
        return;

    const auto i = ni + xi * n;
    Real& v_ = v[i];
    const Real& pay_ = pay[i];
    if (v_ < pay_)
        v_ = pay_;
}

// Calc B = [1 - θ dt 𝒜]
template<typename Real>
__global__
void
fillB_cuda(
    const size_t n,
    const size_t xDim,
    const size_t ti,
    const Real theta,

    const Real* t,
    const Real* x,
    const Real* a0,
    const Real* ax,
    const Real* axx,

    Real* bl,
    Real* b,
    Real* bu)
{
    // pde index
    size_t ni = blockIdx.x * blockDim.x + threadIdx.x;
    if (ni >= n)
        return;

    // x-record index
    size_t xi = blockIdx.y * blockDim.y + threadIdx.y;
    if (xi >= xDim)
        return;

    auto i = ni + xi * n;
    auto j = ni + ti * n;

    auto dt = t[j + n] - t[j];
    // FIXME: Use xi for dx
    auto dx = x[ni + 1 * n] - x[ni + 0 * n];
    auto inv_dx = 1. / dx;
    auto inv_dx2 = inv_dx * inv_dx;

    if (xi == 0) {
        bl[i] = 0;
        b[i] = 1 - theta * dt * (a0[j] - ax[j] * inv_dx);
        bu[i] = -theta * dt * (ax[j] * inv_dx);
    }
    else if (xi < xDim - 1) {
        bl[i] = -theta * dt * (axx[j] * inv_dx2 - 0.5 * ax[j] * inv_dx);
        b[i] = 1 - theta * dt * (a0[j] - 2 * axx[j] * inv_dx2);
        bu[i] = -theta * dt * (axx[j] * inv_dx2 + 0.5 * ax[j] * inv_dx);
    }
    else {
        bl[i] = -theta * dt * (-ax[j] * inv_dx);
        b[i] = 1 - theta * dt * (a0[j] + ax[j] * inv_dx);
        bu[i] = 0;
    }
};


template<typename Real>
__global__
void
fillW_cuda(
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
    Real* w)
{
    // pde index
    size_t ni = blockIdx.x * blockDim.x + threadIdx.x;
    if (ni >= n)
        return;

    // x-record index
    size_t xi = blockIdx.y * blockDim.y + threadIdx.y;
    if (xi >= xDim)
        return;

    auto i = ni + xi * n;
    auto j = ni + ti * n;

    auto dt = (1 - theta) * (t[j + 1 * n] - t[j + 0 * n]);
    auto inv_dx = 1. / (x[ni + n] - x[ni]);
    auto inv_dx2 = inv_dx * inv_dx;

    if (xi == 0)
    {
        w[i] = (1 + dt * a0[j]) * v[i] + dt * (ax[j] * inv_dx) * (v[i + n] - v[i]);
    }
    else if (xi < xDim - 1) {
        w[i] = (1 + dt * a0[j]) * v[i] +
            dt * (0.5 * ax[j] * inv_dx) * (v[i + n] - v[i - n]) +
            dt * (axx[j] * inv_dx2) * (v[i + n] - 2 * v[i] + v[i - n]);
    }
    else {
        w[i] = (1 + dt * a0[j]) * v[i] + dt * (ax[j] * inv_dx) * (v[i] - v[i - n]);
    }
}


template<typename Real>
Error
adjustEarlyExercise(
    const dim3 block2d,
    const size_t bDim,
    const size_t xDim,
    const Real* pay,
    Real* v)
{
    dim3 grid2d(
        (bDim + block2d.x - 1) / block2d.x,
        (xDim + block2d.y - 1) / block2d.y);

    adjustEarlyExercise_cuda<<<grid2d, block2d>>>(bDim, xDim , pay, v);
    cudaDeviceSynchronize();

    return "";
}

template<typename Real>
Error
fillB(
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
    Real* bl,
    Real* b,
    Real* bu)
{
    dim3 grid2d(
        (n + block2d.x - 1) / block2d.x,
        (xDim + block2d.y - 1) / block2d.y);

    fillB_cuda<<<grid2d, block2d>>>(n, xDim, ti, theta, t, x, a0, ax, axx, bl, b, bu);
    cudaDeviceSynchronize();

    return "";
}

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
    Real* w)
{
    dim3 grid2d(
        (n + block2d.x - 1) / block2d.x,
        (xDim + block2d.y - 1) / block2d.y);

    fillW_cuda<<<grid2d, block2d>>>(n, xDim, ti, theta, t, x, a0, ax, axx, v, w);
    cudaDeviceSynchronize();

    return "";
}


template
kw::Error
adjustEarlyExercise(
    const dim3 block2d,
    const size_t n,
    const size_t xDim,
    const float* pay,
    float* v);

template
kw::Error
adjustEarlyExercise(
    const dim3 block2d,
    const size_t n,
    const size_t xDim,
    const double* pay,
    double* v);


template
kw::Error
kw::fillB(
    const dim3 block2d,
    const size_t n,
    const size_t xDim,
    const size_t ti,
    const float theta,
    const float* t,
    const float* x,
    const float* a0,
    const float* ax,
    const float* axx,
    float* bl,
    float* b,
    float* bu);

template
kw::Error
kw::fillB(
    const dim3 block2d,
    const size_t n,
    const size_t xDim,
    const size_t ti,
    const double theta,
    const double* t,
    const double* x,
    const double* a0,
    const double* ax,
    const double* axx,
    double* bl,
    double* b,
    double* bu);


template
kw::Error
kw::fillW(
    const dim3 block2d,
    const size_t n,
    const size_t xDim,
    const size_t ti,
    const float theta,
    const float* t,
    const float* x,
    const float* a0,
    const float* ax,
    const float* axx,
    const float* v,
    float* w);

template
kw::Error
kw::fillW(
    const dim3 block2d,
    const size_t n,
    const size_t xDim,
    const size_t ti,
    const double theta,
    const double* t,
    const double* x,
    const double* a0,
    const double* ax,
    const double* axx,
    const double* v,
    double* w);

}
