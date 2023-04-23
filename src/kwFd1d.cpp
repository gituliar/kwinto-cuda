#pragma comment(lib, "cusparse.lib")

#include "kwFd1d.h"

#include "cuda.h"
#include "cusparse.h"

#define KW_BENCHMARK_ON
#include "kwBenchmark.h"



template<typename Real>
kw::Error
kw::Fd1d<Real>::allocate(const Fd1dConfig& config)
{
    const auto& n = config.pdeCount;
    const auto& tDim = config.tDim;
    const auto& xDim = config.xDim;

    m_theta = config.theta;

    m_cap = n;
    m_tDim = tDim;
    m_xDim = xDim;

    Error error;
    if (error = m_x.init(n, xDim); !error.empty())
        goto lError;
    if (error = m_bl.init(n, xDim); !error.empty())
        goto lError;
    if (error = m_b.init(n, xDim); !error.empty())
        goto lError;
    if (error = m_bu.init(n, xDim); !error.empty())
        goto lError;
    if (error = m_v.init(n, xDim); !error.empty())
        goto lError;
    if (error = m_w.init(n, xDim); !error.empty())
        goto lError;

    if (error = m_t.init(n, tDim); !error.empty())
        goto lError;
    if (error = m_a0.init(n, tDim); !error.empty())
        goto lError;
    if (error = m_ax.init(n, tDim); !error.empty())
        goto lError;
    if (error = m_axx.init(n, tDim); !error.empty())
        goto lError;

    return "";

lError:
    return "kw::Fd1d::allocate: " + error;
};

template<typename Real>
kw::Error
kw::Fd1d<Real>::free()
{
    m_x.free();
    m_bl.free();
    m_b.free();
    m_bu.free();
    m_v.free();
    m_w.free();

    m_t.free();
    m_a0.free();
    m_ax.free();
    m_axx.free();

    m_cap = 0;
    m_tDim = 0;
    m_xDim = 0;

    return "";
}


template<typename Real>
kw::Error
kw::Fd1d<Real>::solve(const std::vector<Fd1dPde<Real>>& pdes)
{
    if (pdes.size() > m_cap)
        return "kw::Fd1d::solve: Not enough memory allocated";

    m_n = pdes.size();

    // 1. Init T-Grid with PDE coefficients
    for (auto i = 0; i < m_n; ++i)
    {
        const auto& pde = pdes[i];

        Real tMin = 0.0, tMax = pde.t;
        Real dt = (tMax - tMin) / (m_tDim - 1);

        for (auto j = 0; j < m_tDim; ++j)
        {
            m_t(i, j) = tMin + j * dt;

            pde.abc(pde.r, pde.q, pde.z, m_a0(i, j), m_ax(i, j), m_axx(i, j));
        }
    }

    // 2. Init X-Grid with Boundary Conditions
    for (auto i = 0; i < m_n; ++i)
    {
        const auto& pde = pdes[i];

        Real xMax = std::max<Real>(log(pde.k) + 10 * pde.z * sqrt(pde.t), log(2.5 * pde.k));
        Real xMin = std::min<Real>(log(pde.k) - 10 * pde.z * sqrt(pde.t), log(0.1 * pde.k));
        Real dx = (xMax - xMin) / (m_xDim - 1);

        for (auto j = 0; j < m_xDim; ++j)
        {
            Real xj = xMin + j * dx;

            pde.payoff(xj, pde.k, m_v(i, j));
            m_x(i, j) = xj;
        }
    }

    for (auto ni = 0; ni < m_n; ++ni)
    {
        if (auto error = solve(pdes, ni); !error.empty())
            return "Fd1d<Real>::solve: " + error;
    }

    return "";
}

template<typename Real>
kw::Error
kw::Fd1d<Real>::solve(const std::vector<Fd1dPde<Real>>& pdes, uint32_t ni)
{
    const auto& pde = pdes[ni];

    for (auto ti = m_tDim-2; ti > 0; --ti)
    {
        // Step 1a
        //   - Calc B = [1 - θ dt 𝒜]

        // NOTE: Assuiming uniform dt & dx
        Real dt = m_t(ni, 1) - m_t(ni, 0);
        Real dx = m_x(ni, 1) - m_x(ni, 0);
        Real inv_dx = static_cast<Real>(1.) / dx;
        Real inv_dx2 = inv_dx * inv_dx;

        auto j = m_t.index(ni, ti);

        {
            auto i = m_x.index(ni, 0);
            m_bl[i] = 0;
            m_b[i] = 1 - m_theta * dt * (m_a0[j] - m_ax[j] * inv_dx);
            m_bu[i] = -m_theta * dt * (m_ax[j] * inv_dx);
        }

        for (auto xi = 1; xi < m_xDim-1; ++xi)
        {
            auto i = m_x.index(ni, xi);
            m_bl[i] = -m_theta * dt * (m_axx[j] * inv_dx2 - m_ax[j] * inv_dx / 2);
            m_b[i] = 1 - m_theta * dt * (m_a0[j] - 2 * m_axx[j] * inv_dx2);
            m_bu[i] = -m_theta * dt * (m_axx[j] * inv_dx2 + m_ax[j] * inv_dx / 2);
        }

        {
            auto i = m_x.index(ni, m_xDim - 1);
            m_bl[i] = -m_theta * dt * (-m_ax[j] * inv_dx);
            m_b[i] = 1 - m_theta * dt * (m_a0[j] + m_ax[j] * inv_dx);
            m_bu[i] = 0;
        }

        // Step 1b
        //   - Calc W = [1 + (1 - θ) dt 𝒜] V(t + dt)
        {
            auto xi = 0;
            auto i = m_x.index(ni, xi);
            m_w[i] = (1 + (1 - m_theta) * dt * m_a0[j]) * m_v[i] +
                (1 - m_theta) * dt * (m_ax[j] * inv_dx) * (m_v(ni, xi + 1) - m_v(ni, xi));
        }

        for (auto xi = 1; xi < m_xDim - 1; ++xi)
        {
            auto i = m_x.index(ni, xi);
            m_w[i] = (1 + (1 - m_theta) * dt * m_a0[j]) * m_v[i] +
                (1 - m_theta) * dt * (m_ax[j] * inv_dx / 2) * (m_v(ni, xi + 1) - m_v(ni, xi - 1)) +
                (1 - m_theta) * dt * (m_axx[j] * inv_dx2) * (m_v(ni, xi + 1) - 2 * m_v[i] + m_v(ni, xi - 1));
        }

        {
            auto xi = m_xDim - 1;
            auto i = m_x.index(ni, xi);
            m_w[i] = (1 + (1 - m_theta) * dt * m_a0[j]) * m_v[i] +
                (1 - m_theta) * dt * (m_ax[j] * inv_dx) * (m_v(ni, xi) - m_v(ni, xi - 1));
        }

        // Step 2
        //   - Solve B V = W
        {
            auto i = m_v.index(ni, 0);
            solveTridiagonal(
                static_cast<int>(m_xDim),
                &m_bl[i],
                &m_b[i],
                &m_bu[i],
                &m_w[i],
                &m_v[i]
            );
        }

        // Step 3
        //  - Adjust for early exercise
        if (pde.earlyExercise)
        {
            Real pay;
            for (auto xi = 0; xi < m_xDim - 1; ++xi)
            {
                pde.payoff(m_x(ni, xi), pde.k, pay);
                m_v(ni, xi) = std::max(m_v(ni, xi), pay);
            }
        }
    }

    return "";
}


template<typename Real>
kw::Error
kw::Fd1d<Real>::value(const size_t ni, const Real s, Real& v) const
{
    Real x = std::log(s);

    if (ni >= m_n)
        return "Fd1d::value: Solution index out of range";

    size_t xi = 0;
    while ((xi < m_xDim) && (m_x(ni, xi) < x))
        ++xi;

    if ((xi == 0) || (xi == m_xDim))
    {
        const auto sMin = std::exp(m_x(ni, 0));
        const auto sMax = std::exp(m_x(ni, m_xDim - 1));
        return "Fd1d::value: Spot " + std::to_string(s) + " not in range (" + std::to_string(sMin)
            + ", " + std::to_string(sMax) + ")";
    }

    v = ((m_x(ni, xi) - x) * m_v(ni, xi - 1) + (x - m_x(ni, xi - 1)) * m_v(ni, xi)) / (m_x(ni, xi) - m_x(ni, xi - 1));

    return "";
}



template<typename Real>
kw::Error
kw::Fd1d_Gpu<Real>::allocate(const Fd1dConfig& config)
{
    Error error;

    const auto& n = config.pdeCount;
    const auto& tDim = config.tDim;
    const auto& xDim = config.xDim;

    m_nThreads = config.nThreads;
    m_xThreads = config.xThreads;

    m_cap = n;
    m_tDim = tDim;
    m_xDim = xDim;

    if (error = m__v.init(n, xDim); !error.empty()) goto cudaError;
    if (error = m__x.init(n, xDim); !error.empty()) goto cudaError;

    if (error = m__t.init(n, m_tDim); !error.empty()) goto cudaError;
    if (error = m__a0.init(n, m_tDim); !error.empty()) goto cudaError;
    if (error = m__ax.init(n, m_tDim); !error.empty()) goto cudaError;
    if (error = m__axx.init(n, m_tDim); !error.empty()) goto cudaError;
    if (error = m__pay.init(n, m_xDim); !error.empty()) goto cudaError;

    if (error = m_x.init(n, xDim); !error.empty()) goto cudaError;
    if (error = m_bl.init(n, xDim); !error.empty()) goto cudaError;
    if (error = m_b.init(n, xDim); !error.empty()) goto cudaError;
    if (error = m_bu.init(n, xDim); !error.empty()) goto cudaError;
    if (error = m_v.init(n, xDim); !error.empty()) goto cudaError;
    if (error = m_w.init(n, xDim); !error.empty()) goto cudaError;

    if (error = m_pay.init(n, xDim); !error.empty()) goto cudaError;

    if (error = m_t.init(n, tDim); !error.empty()) goto cudaError;
    if (error = m_a0.init(n, tDim); !error.empty()) goto cudaError;
    if (error = m_ax.init(n, tDim); !error.empty()) goto cudaError;
    if (error = m_axx.init(n, tDim); !error.empty()) goto cudaError;


    if (auto status = cusparseCreate(&m_cusparseHandle); status != CUSPARSE_STATUS_SUCCESS)
        return "kw::Fd1d_Gpu::allocate: cuSparse error " + status;

    m_cusparseBufSize = 512 * 1024;
    if (auto cuErr = cudaMalloc((void**)&m_cusparseBuf, m_cusparseBufSize); cuErr != cudaSuccess)
        goto cudaError;

    return "";

cudaError:
    return "kw::Fd1d_Gpu::allocate: " + error;
}

template<typename Real>
kw::Error
kw::Fd1d_Gpu<Real>::free()
{
    if (auto status = cusparseDestroy(m_cusparseHandle); status != CUSPARSE_STATUS_SUCCESS)
        return "kw::Fd1d_Gpu::allocate: cuSparse error " + status;

    m__v.free();
    m__x.free();

    m__t.free();
    m__a0.free();
    m__ax.free();
    m__axx.free();
    m__pay.free();

    m_x.free();
    m_bl.free();
    m_b.free();
    m_bu.free();
    m_v.free();
    m_w.free();
    m_pay.free();

    m_t.free();
    m_a0.free();
    m_ax.free();
    m_axx.free();

    m_cap = 0;
    m_tDim = 0;
    m_xDim = 0;

    if (m_cusparseBuf)
    {
        cudaFree(m_cusparseBuf);
        m_cusparseBufSize = 0;
    }
    return "";
}


template<typename Real>
kw::Error
kw::Fd1d_Gpu<Real>::solve(const std::vector<Fd1dPde<Real>>& pdes)
{
    if (pdes.size() > m_cap)
        return "kw::Fd1d_Gpu::init: Not enough memory allocated";

    m_n = pdes.size();

    m_theta = 0.5; // pde.theta;

    // 1. Init T-Grid with PDE coefficients
    {
        for (auto i = 0; i < m_n; ++i)
        {
            const auto& pde = pdes[i];

            Real tMin = 0.0, tMax = pde.t;
            Real dt = (tMax - tMin) / (m_tDim - 1);

            for (auto j = 0; j < m_tDim; ++j)
            {
                m__t(i, j) = tMin + j * dt;

                pde.abc(pde.r, pde.q, pde.z, m__a0(i, j), m__ax(i, j), m__axx(i, j));
            }
        }


        // CPU -> GPU
        m_t = m__t;
        m_a0 = m__a0;
        m_ax = m__ax;
        m_axx = m__axx;
    }

    // 2. Init X-Grid with Boundary Conditions
    {
        for (auto ni = 0; ni < m_n; ++ni)
        {
            const auto& pde = pdes[ni];

            Real xMax = log(pde.k) + 5 * pde.z * sqrt(pde.t);
            Real xMin = log(pde.k) - 5 * pde.z * sqrt(pde.t);
            Real dx = (xMax - xMin) / (m_xDim - 1);

            for (auto xi = 0; xi < m_xDim; ++xi)
            {
                Real x = xMin + xi * dx;

                pde.payoff(x, pde.k, m__v(ni, xi));
                m__x(ni, xi) = x;

                m__pay(ni, xi) = pde.earlyExercise ? m__v(ni, xi) : 0;
            }
        }

        // CPU -> GPU
        {
            m_pay = m__pay;
            m_v = m__v;
            m_x = m__x;
        }
    }

    cusparseStatus_t status;

    for (auto ti = m_tDim-2; ti > 0; --ti)
    {
        // Step 1a
        //   - Calc B = [1 - θ dt 𝒜]
        fillB(m_nThreads, m_xThreads, ti, m_theta, m_t, m_a0, m_ax, m_axx, m_x, m_bl, m_b, m_bu);

        // Step 1b
        //   - Calc W = [1 + (1 - θ) dt 𝒜] V(t + dt)
        fillW(m_nThreads, m_xThreads, ti, m_theta, m_t, m_a0, m_ax, m_axx, m_x, m_v, m_w);

        // Step 2a
        //   - Solve B X = W (places X in W)
        int constexpr algo = 0;
        if (ti == 0)
        {
            size_t bufSize;
            if constexpr (std::is_same_v<Real, float>)
            {
                status = cusparseSgtsvInterleavedBatch_bufferSizeExt(
                    m_cusparseHandle,
                    algo,
                    static_cast<int>(m_xDim),
                    &m_bl[0],
                    &m_b[0],
                    &m_bu[0],
                    &m_w[0],
                    static_cast<int>(m_n),
                    &bufSize
                );
            }
            else if constexpr (std::is_same_v<Real, double>)
            {
                status = cusparseDgtsvInterleavedBatch_bufferSizeExt(
                    m_cusparseHandle,
                    algo,
                    static_cast<int>(m_xDim),
                    &m_bl[0],
                    &m_b[0],
                    &m_bu[0],
                    &m_w[0],
                    static_cast<int>(m_n),
                    &bufSize
                );
            }
            else
                static_assert(true, "type not supported");
            if (status != CUSPARSE_STATUS_SUCCESS)
                return "kw::Fd1d_Gpu::solve: cuSparse error " + status;

            if (bufSize > m_cusparseBufSize)
            {
                cudaFree(m_cusparseBuf);

                m_cusparseBufSize = bufSize;
                if (auto cuErr = cudaMalloc((void**)&m_cusparseBuf, m_cusparseBufSize); cuErr != CUDA_SUCCESS)
                    return "kw::Fd1d_Gpu::solve: " + fromCudaError(cuErr);
            }
        }

        if constexpr (std::is_same_v<Real, float>)
        {
            status = cusparseSgtsvInterleavedBatch(
                m_cusparseHandle,
                algo,
                static_cast<int>(m_xDim),
                &m_bl[0],
                &m_b[0],
                &m_bu[0],
                &m_w[0],
                static_cast<int>(m_n),
                m_cusparseBuf
            );
        }
        else if constexpr (std::is_same_v<Real, double>)
        {
            status = cusparseDgtsvInterleavedBatch(
                m_cusparseHandle,
                algo,
                static_cast<int>(m_xDim),
                &m_bl[0],
                &m_b[0],
                &m_bu[0],
                &m_w[0],
                static_cast<int>(m_n),
                m_cusparseBuf
            );
        }
        else
            static_assert(true, "type not supported");
        if (status != CUSPARSE_STATUS_SUCCESS)
            return "kw::Fd1d_Gpu::solve: cuSparse error " + status;

        // Step 2b
        //   - Swap V & W
        {
            auto vw = std::move(m_v);
            m_v = m_w;
            m_w = vw;
        }

        // Step 3
        //  - Adjust for early exercise
        adjustEarlyExercise(m_nThreads, m_xThreads, m_pay, m_v);
    }

    m__v = m_v;

    return "";
}

template<typename Real>
kw::Error
kw::Fd1d_Gpu<Real>::value(const size_t ni, const Real s, Real& v) const
{
    Real x = std::log(s);

    if (ni >= m_n)
        return "Fd1d_Gpu::value: Solution index out of range";

    size_t xi = 0;
    while ((xi < m_xDim) && (m__x(ni, xi) < x))
        ++xi;

    if ((xi == 0) || (xi == m_xDim))
    {
        const auto sMin = std::exp(m_x(ni, 0));
        const auto sMax = std::exp(m_x(ni, m_xDim - 1));
        return "Fd1d_Gpu::value: Spot " + std::to_string(s) + " not in range (" + std::to_string(sMin)
            + ", " + std::to_string(sMax) + ")";
    }

    v = ((m__x(ni, xi) - x) * m__v(ni, xi - 1) + (x - m__x(ni, xi - 1)) * m__v(ni, xi)) / (m__x(ni, xi) - m__x(ni, xi - 1));

    return "";
}

template class kw::Fd1d<double>;
template class kw::Fd1d<float>;

template class kw::Fd1d_Gpu<double>;
template class kw::Fd1d_Gpu<float>;
