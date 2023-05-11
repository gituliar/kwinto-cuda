#include "kwFd1d.h"

#include "cuda.h"
#include "cusparse.h"


template<typename Real>
kw::Error
kw::Fd1d<Real>::init(size_t bCap, size_t tDim, size_t xDim)
{
    //m_theta = config.get("FD1D.THETA", 0.5);
    m_theta = 0.5;

    if (auto error = m_bl.init(bCap, xDim); !error.empty())
        return "Fd1d::init: " + error;
    if (auto error = m_b.init(bCap, xDim); !error.empty())
        return "Fd1d::init: " + error;
    if (auto error = m_bu.init(bCap, xDim); !error.empty())
        return "Fd1d::init: " + error;
    if (auto error = m_v.init(bCap, xDim); !error.empty())
        return "Fd1d::init: " + error;
    if (auto error = m_w.init(bCap, xDim); !error.empty())
        return "Fd1d::init: " + error;

    if (auto error = m_a0.init(bCap, tDim); !error.empty())
        return "Fd1d::init: " + error;
    if (auto error = m_ax.init(bCap, tDim); !error.empty())
        return "Fd1d::init: " + error;
    if (auto error = m_axx.init(bCap, tDim); !error.empty())
        return "Fd1d::init: " + error;

    return "";
};

template<typename Real>
kw::Error
kw::Fd1d<Real>::free()
{
    m_bl.free();
    m_b.free();
    m_bu.free();
    m_v.free();
    m_w.free();

    m_a0.free();
    m_ax.free();
    m_axx.free();

    return "";
}


template<typename Real>
kw::Error
kw::Fd1d<Real>::solve(
    const std::vector<Fd1dPde<Real>>& batch,
    const CpuGrid& tGrid,
    const CpuGrid& xGrid,
    const CpuGrid& vGrid)
{
    if (batch.size() > m_v.cols())
        return "Fd1d::solve: Not enough memory allocated";

    // 1. Init T-Grid with PDE coefficients
    for (auto bi = 0; bi < batch.size(); ++bi) {
        const auto& pde = batch[bi];

        for (auto ti = 0; ti < tGrid.rows(); ++ti) {
            m_a0(bi, ti) = pde.a0;
            m_ax(bi, ti) = pde.ax;
            m_axx(bi, ti) = pde.axx;
        }
    }

    // 2. Init X-Grid with Boundary Conditions
    for (auto bi = 0; bi < batch.size(); ++bi) {
        for (auto xi = 0; xi < vGrid.rows(); ++xi)
            m_v(bi, xi) = vGrid(bi, xi);
    }

    for (auto bi = 0; bi < batch.size(); ++bi) {
        if (auto error = solveOne(bi, tGrid, xGrid, vGrid); !error.empty())
            return "Fd1d<Real>::solve: " + error;
    }

    return "";
}

template<typename Real>
kw::Error
kw::Fd1d<Real>::solveOne(
    const uint32_t ni,
    const CpuGrid& tGrid,
    const CpuGrid& xGrid,
    const CpuGrid& vGrid)
{
    const auto bCap = tGrid.cols();
    const auto tDim = tGrid.rows();
    const auto xDim = xGrid.rows();

    for (int ti = tDim - 2; ti >= 0; --ti) {
        // Step 1
        //   - Calc B = [1 - θ dt 𝒜]
        //   - Calc W = [1 + (1 - θ) dt 𝒜] V(t + dt)

        Real dt = tGrid(ni, ti + 1) - tGrid(ni, ti);
        {
            const auto xi = 0;
            const Real inv_dx = static_cast<Real>(1.) / (xGrid(ni, xi + 1) - xGrid(ni, xi));

            m_bl(ni, xi) = 0;
            m_b(ni, xi) = 1 - m_theta * dt * (m_a0(ni, ti) - inv_dx * m_ax(ni, ti));
            m_bu(ni, xi) = -m_theta * dt * (inv_dx * m_ax(ni, ti));

            m_w(ni, xi) = (1 + (1 - m_theta) * dt * m_a0(ni, ti)) * m_v(ni, xi) +
                (1 - m_theta) * dt * (m_ax(ni, ti) * inv_dx) * (m_v(ni, xi + 1) - m_v(ni, xi));
        }

        for (auto xi = 1; xi < xDim - 1; ++xi) {
            const Real inv_dx = static_cast<Real>(1.) / (xGrid(ni, xi) - xGrid(ni, xi - 1));
            const Real inv_dx2 = inv_dx * inv_dx;

            m_bl(ni, xi) = -m_theta * dt * (inv_dx2 * m_axx(ni, ti) - inv_dx / 2 * m_ax(ni, ti));
            m_b(ni, xi) = 1 - m_theta * dt * (m_a0(ni, ti) - 2 * inv_dx2 * m_axx(ni, ti));
            m_bu(ni, xi) = -m_theta * dt * (inv_dx2 * m_axx(ni, ti) + inv_dx / 2 * m_ax(ni, ti));

            m_w(ni, xi) = (1 + (1 - m_theta) * dt * m_a0(ni, ti)) * m_v(ni, xi) +
                (1 - m_theta) * dt * (m_ax(ni, ti) * inv_dx / 2) * (m_v(ni, xi + 1) - m_v(ni, xi - 1)) +
                (1 - m_theta) * dt * (m_axx(ni, ti) * inv_dx2) * (m_v(ni, xi + 1) - 2 * m_v(ni, xi) + m_v(ni, xi - 1));
        }

        {
            const auto xi = xDim - 1;
            const Real inv_dx = static_cast<Real>(1.) / (xGrid(ni, xi) - xGrid(ni, xi - 1));

            m_bl(ni, xDim - 1) = -m_theta * dt * (-inv_dx * m_ax(ni, ti));
            m_b(ni, xDim - 1) = 1 - m_theta * dt * (m_a0(ni, ti) + inv_dx * m_ax(ni, ti));
            m_bu(ni, xDim - 1) = 0;

            m_w(ni, xDim - 1) = (1 + (1 - m_theta) * dt * m_a0(ni, ti)) * m_v(ni, xi) +
                (1 - m_theta) * dt * (m_ax(ni, ti) * inv_dx) * (m_v(ni, xi) - m_v(ni, xi - 1));
        }

        // Step 2
        //   - Solve B V = W
        solveTridiagonal(
            static_cast<int>(xDim),
            &m_bl(ni, 0),
            &m_b(ni, 0),
            &m_bu(ni, 0),
            &m_w(ni, 0),
            &m_v(ni, 0),
            bCap
        );

        // Step 3
        //  - Adjust for early exercise
        for (auto xi = 0; xi < xDim - 1; ++xi)
            m_v(ni, xi) = std::max(m_v(ni, xi), vGrid(ni, xi));
    }

    return "";
}


template<typename Real>
kw::Error
kw::Fd1d<Real>::value(
    const size_t ni,
    const Real s,
    const CpuGrid& xGrid,
    Real& v) const
{
    Real x = std::log(s);

    const auto bCap = xGrid.cols();
    const auto xDim = xGrid.rows();

    if (ni >= bCap)
        return "Fd1d::value: Solution index out of range";

    size_t xi = 0;
    while ((xi < xDim) && (xGrid(ni, xi) < x))
        ++xi;

    if ((xi == 0) || (xi == xDim)) {
        const auto sMin = std::exp(xGrid(ni, 0));
        const auto sMax = std::exp(xGrid(ni, xDim - 1));
        return "Fd1d::value: Spot " + std::to_string(s) + " not in range (" + std::to_string(sMin)
            + ", " + std::to_string(sMax) + ")";
    }

    v = ((xGrid(ni, xi) - x) * m_v(ni, xi - 1) + (x - xGrid(ni, xi - 1)) * m_v(ni, xi)) / (xGrid(ni, xi) - xGrid(ni, xi - 1));

    return "";
}



template<typename Real>
kw::Error
kw::Fd1d_Gpu<Real>::init(const Config& config, size_t tDim, size_t xDim)
{
    const auto bCap = config.get("FD1D.BATCH_SZIE", 64);

    m_theta = config.get("FD1D.THETA", 0.5);

    m_nThreads = config.get("FD1D.N_THREADS", 1);
    m_xThreads = config.get("FD1D.X_THREADS", 64);


    if (auto error = m_v.init(bCap, xDim); !error.empty())
        return "Fd1d_Gpu::init: " + error;

    if (auto error = m_a0.init(bCap, tDim); !error.empty())
        return "Fd1d_Gpu::init: " + error;
    if (auto error = m_ax.init(bCap, tDim); !error.empty())
        return "Fd1d_Gpu::init: " + error;
    if (auto error = m_axx.init(bCap, tDim); !error.empty())
        return "Fd1d_Gpu::init: " + error;

    if (auto error = m__x.init(bCap, xDim); !error.empty())
        return "Fd1d_Gpu::init: " + error;
    if (auto error = m__bl.init(bCap, xDim); !error.empty())
        return "Fd1d_Gpu::init: " + error;
    if (auto error = m__b.init(bCap, xDim); !error.empty())
        return "Fd1d_Gpu::init: " + error;
    if (auto error = m__bu.init(bCap, xDim); !error.empty())
        return "Fd1d_Gpu::init: " + error;
    if (auto error = m__v.init(bCap, xDim); !error.empty())
        return "Fd1d_Gpu::init: " + error;
    if (auto error = m__w.init(bCap, xDim); !error.empty())
        return "Fd1d_Gpu::init: " + error;

    if (auto error = m__pay.init(bCap, xDim); !error.empty())
        return "Fd1d_Gpu::init: " + error;

    if (auto error = m__t.init(bCap, tDim); !error.empty())
        return "Fd1d_Gpu::init: " + error;
    if (auto error = m__a0.init(bCap, tDim); !error.empty())
        return "Fd1d_Gpu::init: " + error;
    if (auto error = m__ax.init(bCap, tDim); !error.empty())
        return "Fd1d_Gpu::init: " + error;
    if (auto error = m__axx.init(bCap, tDim); !error.empty())
        return "Fd1d_Gpu::init: " + error;


    if (auto status = cusparseCreate(&m_cusparseHandle); status != CUSPARSE_STATUS_SUCCESS)
        return "Fd1d_Gpu::free: cusparseStatus = " + std::to_string(status);

    m_cusparseBufSize = 512 * 1024;
    if (auto cuErr = cudaMalloc((void**)&m_cusparseBuf, m_cusparseBufSize); cuErr != cudaSuccess)
        return "Fd1d_Gpu::init: cudaError = " + std::to_string(cuErr);

    return "";
}

template<typename Real>
kw::Error
kw::Fd1d_Gpu<Real>::free()
{
    if (auto status = cusparseDestroy(m_cusparseHandle); status != CUSPARSE_STATUS_SUCCESS)
        return "Fd1d_Gpu::free: cusparseStatus = " + std::to_string(status);

    m_v.free();

    m_a0.free();
    m_ax.free();
    m_axx.free();

    m__x.free();
    m__bl.free();
    m__b.free();
    m__bu.free();
    m__v.free();
    m__w.free();
    m__pay.free();

    m__t.free();
    m__a0.free();
    m__ax.free();
    m__axx.free();

    if (m_cusparseBuf) {
        cudaFree(m_cusparseBuf);
        m_cusparseBufSize = 0;
    }
    return "";
}


template<typename Real>
kw::Error
kw::Fd1d_Gpu<Real>::solve(
    const std::vector<Fd1dPde<Real>>& batch,
    const CpuGrid& tGrid,
    const CpuGrid& xGrid,
    const CpuGrid& vGrid)
{
    const auto bCap = tGrid.cols();
    const auto bDim = batch.size();
    const auto tDim = tGrid.rows();
    const auto xDim = xGrid.rows();

    if (bDim > bCap)
        return "Fd1d_Gpu::init: Batch is bigger than capacity";

    // 1. Init PDE coefficients
    {
        m_a0.resize(bDim, tDim);
        m_ax.resize(bDim, tDim);
        m_axx.resize(bDim, tDim);
        for (auto i = 0; i < bDim; ++i) {
            const auto& pde = batch[i];
            for (auto j = 0; j < tDim; ++j) {
                m_a0(i, j) = pde.a0;
                m_ax(i, j) = pde.ax;
                m_axx(i, j) = pde.axx;
            }
        }

        // CPU -> GPU
        m__t = tGrid;
        m__a0 = m_a0;
        m__ax = m_ax;
        m__axx = m_axx;
    }

    // 2. Init X-Grid with Boundary Conditions
    {
        m_v.resize(bDim, xDim);
        m__bl.resize(bDim, xDim);
        m__b.resize(bDim, xDim);
        m__bu.resize(bDim, xDim);
        m__w.resize(bDim, xDim);

        // CPU -> GPU
        m__pay = vGrid;
        m__v = vGrid;
        m__x = xGrid;
    }

    cusparseStatus_t status;
    for (auto ti = tDim - 2; ti > 0; --ti) {
        // Step 1a
        //   - Calc B = [1 - θ dt 𝒜]
        fillB(m_nThreads, m_xThreads, ti, m_theta, m__t, m__x, m__a0, m__ax, m__axx, m__bl, m__b, m__bu);

        // Step 1b
        //   - Calc W = [1 + (1 - θ) dt 𝒜] V(t + dt)
        fillW(m_nThreads, m_xThreads, ti, m_theta, m__t, m__x, m__a0, m__ax, m__axx, m__v, m__w);

        // Step 2a
        //   - Solve B X = W (places X in W)
        int constexpr algo = 0;
        if (ti == tDim - 2) {
            size_t bufSize;
            if constexpr (std::is_same_v<Real, float>) {
                status = cusparseSgtsvInterleavedBatch_bufferSizeExt(
                    m_cusparseHandle,
                    algo,
                    static_cast<int>(xDim),
                    &m__bl[0],
                    &m__b[0],
                    &m__bu[0],
                    &m__w[0],
                    static_cast<int>(bCap),
                    &bufSize
                );
            }
            else if constexpr (std::is_same_v<Real, double>) {
                status = cusparseDgtsvInterleavedBatch_bufferSizeExt(
                    m_cusparseHandle,
                    algo,
                    static_cast<int>(xDim),
                    &m__bl[0],
                    &m__b[0],
                    &m__bu[0],
                    &m__w[0],
                    static_cast<int>(bCap),
                    &bufSize
                );
            }
            //else {
            //    static_assert(false, "type not supported");
            //}
            if (status != CUSPARSE_STATUS_SUCCESS)
                return "Fd1d_Gpu::solve: cusparseStatus = " + std::to_string(status);

            if (bufSize > m_cusparseBufSize) {
                cudaFree(m_cusparseBuf);

                m_cusparseBufSize = bufSize;
                if (auto cuErr = cudaMalloc((void**)&m_cusparseBuf, m_cusparseBufSize); cuErr != cudaSuccess)
                    return "Fd1d_Gpu::solve: cudaError = " + fromCudaError(cuErr);
            }
        }

        if constexpr (std::is_same_v<Real, float>) {
            status = cusparseSgtsvInterleavedBatch(
                m_cusparseHandle,
                algo,
                static_cast<int>(xDim),
                &m__bl[0],
                &m__b[0],
                &m__bu[0],
                &m__w[0],
                static_cast<int>(bCap),
                m_cusparseBuf
            );
        }
        else if constexpr (std::is_same_v<Real, double>) {
            status = cusparseDgtsvInterleavedBatch(
                m_cusparseHandle,
                algo,
                static_cast<int>(xDim),
                &m__bl[0],
                &m__b[0],
                &m__bu[0],
                &m__w[0],
                static_cast<int>(bCap),
                m_cusparseBuf
            );
        }
        //else
        //    static_assert(false, "type not supported");

        if (status != CUSPARSE_STATUS_SUCCESS)
            return "Fd1d_Gpu::solve: cusparseStatus = " + std::to_string(status);

        // Step 2b
        //   - Swap V & W
        {
            const auto vw = m__v;
            m__v = m__w;
            m__w = vw;
        }

        // Step 3
        //  - Adjust for early exercise
        adjustEarlyExercise(m_nThreads, m_xThreads, m__pay, m__v);
    }

    m_v = m__v;

    return "";
}

template<typename Real>
kw::Error
kw::Fd1d_Gpu<Real>::value(
    const size_t ni,
    const Real s,
    const CpuGrid& xGrid,
    Real& v) const
{
    Real x = std::log(s);

    const auto bCap = xGrid.cols();
    const auto xDim = xGrid.rows();

    if (ni >= bCap)
        return "Fd1d::value: Solution index out of range";

    size_t xi = 0;
    while ((xi < xDim) && (xGrid(ni, xi) < x))
        ++xi;

    if ((xi == 0) || (xi == xDim)) {
        const auto sMin = std::exp(xGrid(ni, 0));
        const auto sMax = std::exp(xGrid(ni, xDim - 1));
        return "Fd1d_Gpu::value: Spot " + std::to_string(s) + " not in range (" + std::to_string(sMin)
            + ", " + std::to_string(sMax) + ")";
    }

    v = ((xGrid(ni, xi) - x) * m_v(ni, xi - 1) + (x - xGrid(ni, xi - 1)) * m_v(ni, xi)) / (xGrid(ni, xi) - xGrid(ni, xi - 1));

    return "";
}

template class kw::Fd1d<double>;
template class kw::Fd1d<float>;

template class kw::Fd1d_Gpu<double>;
template class kw::Fd1d_Gpu<float>;
