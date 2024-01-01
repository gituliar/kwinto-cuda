#include "kwFd1d.h"

#include <iostream>

#include "Math/kwMath.h"
#include "kwThreadPool.h"

using namespace kw;


Error
Fd1d::init(u64 tDim, u64 xDim)
{
    m_theta = 0.5;

    m_tDim = tDim;
    m_xDim = xDim;

    return "";
};


Error
Fd1d::solve(
    const std::vector<Fd1dPde>& batch,
    const CpuGrid& tGrid,
    const CpuGrid& xGrid,
    const CpuGrid& vGrid)
{
    // 1. Reserve memory
    const auto n = batch.size();
    m_bl.resize(n, m_xDim);
    m_b.resize(n, m_xDim);
    m_bu.resize(n, m_xDim);
    m_v.resize(n, m_xDim);
    m_w.resize(n, m_xDim);

    m_a0.resize(n, m_tDim);
    m_ax.resize(n, m_tDim);
    m_axx.resize(n, m_tDim);

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

    auto priceJob = [&](u64 i) {
        std::cout << i << std::endl;
    };

    auto solveJob = [&](u64 i) {
        if (auto error = solveOne(i, tGrid, xGrid, vGrid); !error.empty()) {
            std::cerr << "Fd1d<Real>::solve: " + error << std::endl;
        }
    };
    auto& pool = kw::ThreadPool::instance();
    for (auto bi = 0; bi < batch.size(); ++bi) {
        pool.add_job(std::bind(solveJob, bi));
    }
    pool.wait();

    return "";
}

Error
Fd1d::solveOne(
    const u64 ni,
    const CpuGrid& tGrid,
    const CpuGrid& xGrid,
    const CpuGrid& vGrid)
{
    for (int ti = m_tDim - 2; ti >= 0; --ti) {
        // Step 1
        //   - Calc B = [1 - θ dt 𝒜]
        //   - Calc W = [1 + (1 - θ) dt 𝒜] V(t + dt)

        f64 dt = tGrid(ni, ti + 1) - tGrid(ni, ti);
        {
            const auto xi = 0;
            //const auto i = ni + xi * n;
            //const auto j = ni + ti * n;

            const f64 inv_dx = 1. / (xGrid(ni, xi + 1) - xGrid(ni, xi));

            m_bl(ni, xi) = 0;
            m_b(ni, xi) = 1 - m_theta * dt * (m_a0(ni, ti) - inv_dx * m_ax(ni, ti));
            m_bu(ni, xi) = -m_theta * dt * (inv_dx * m_ax(ni, ti));

            m_w(ni, xi) = (1 + (1 - m_theta) * dt * m_a0(ni, ti)) * m_v(ni, xi) +
                (1 - m_theta) * dt * (m_ax(ni, ti) * inv_dx) * (m_v(ni, xi + 1) - m_v(ni, xi));
        }

        for (auto xi = 1; xi < m_xDim - 1; ++xi) {
            const f64 inv_dxu = 1. / (xGrid(ni, xi + 1) - xGrid(ni, xi));
            const f64 inv_dxm = 1. / (xGrid(ni, xi + 1) - xGrid(ni, xi - 1));
            const f64 inv_dxd = 1. / (xGrid(ni, xi) - xGrid(ni, xi - 1));

            const f64 inv_dx2u = 2. * inv_dxu * inv_dxm;
            const f64 inv_dx2m = 2. * inv_dxd * inv_dxu;
            const f64 inv_dx2l = 2. * inv_dxd * inv_dxm;

            m_bl(ni, xi) = -m_theta * dt * (-inv_dxm * m_ax(ni, ti) + inv_dx2l * m_axx(ni, ti));
            m_b(ni, xi) = 1 - m_theta * dt * (m_a0(ni, ti) - inv_dx2m * m_axx(ni, ti));
            m_bu(ni, xi) = -m_theta * dt * (inv_dxm * m_ax(ni, ti) + inv_dx2u * m_axx(ni, ti));

            m_w(ni, xi) = (1 + (1 - m_theta) * dt * m_a0(ni, ti)) * m_v(ni, xi) +
                (1 - m_theta) * dt * (m_ax(ni, ti) * inv_dxm) * (m_v(ni, xi + 1) - m_v(ni, xi - 1)) +
                (1 - m_theta) * dt * (m_axx(ni, ti)) * (inv_dx2u * m_v(ni, xi + 1) - inv_dx2m * m_v(ni, xi) + inv_dx2l * m_v(ni, xi - 1));
        }

        {
            const auto xi = m_xDim - 1;
            const f64 inv_dx = 1. / (xGrid(ni, xi) - xGrid(ni, xi - 1));

            m_bl(ni, m_xDim - 1) = -m_theta * dt * (-inv_dx * m_ax(ni, ti));
            m_b(ni, m_xDim - 1) = 1 - m_theta * dt * (m_a0(ni, ti) + inv_dx * m_ax(ni, ti));
            m_bu(ni, m_xDim - 1) = 0;

            m_w(ni, m_xDim - 1) = (1 + (1 - m_theta) * dt * m_a0(ni, ti)) * m_v(ni, xi) +
                (1 - m_theta) * dt * (m_ax(ni, ti) * inv_dx) * (m_v(ni, xi) - m_v(ni, xi - 1));
        }

        // Step 2
        //   - Solve B V = W
        solveTridiagonal(
            static_cast<int>(m_xDim),
            &m_bl(ni, 0),
            &m_b(ni, 0),
            &m_bu(ni, 0),
            &m_w(ni, 0),
            &m_v(ni, 0)
        );

        // Step 3
        //  - Adjust for early exercise
        for (auto xi = 0; xi < m_xDim - 1; ++xi)
            m_v(ni, xi) = std::max(m_v(ni, xi), vGrid(ni, xi));
    }

    return "";
}


Error
Fd1d::value(
    const u64 ni,
    const f64 s,
    const CpuGrid& xGrid,
    f64& v) const
{
    f64 x = std::log(s);

    const auto n = xGrid.cols();
    const auto xDim = xGrid.rows();

    if (ni >= n)
        return "Fd1d::value: Solution index out of range";

    u64 xi = 0;
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
