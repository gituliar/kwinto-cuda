#include "kwFd1d.h"

#include <iostream>

#include "Math/kwMath.h"
#include "kwThreadPool.h"

using namespace kw;


Error
Fd1d::solve(const vector<Fd1dPde>& pdes, const Grid2d& t, const Grid2d& x, const Grid2d& v)
{
    auto n = pdes.size();

    auto tDim = t.rows();
    auto xDim = x.rows();

    m_bl.resize(n, xDim);
    m_b.resize(n, xDim);
    m_bu.resize(n, xDim);
    m_v.resize(n, xDim);
    m_w.resize(n, xDim);

    m_a0.resize(n, tDim);
    m_ax.resize(n, tDim);
    m_axx.resize(n, tDim);

    for (auto i = 0; i < pdes.size(); i++) {
        /// Fill PDE coefficients
        ///
        for (auto ti = 0; ti < t.rows(); ++ti) {
            m_a0(i, ti) = pdes[i].a0;
            m_ax(i, ti) = pdes[i].ax;
            m_axx(i, ti) = pdes[i].axx;
        }

        /// Fill boundary conditions
        ///
        for (auto xi = 0; xi < v.rows(); ++xi)
            m_v(i, xi) = v(i, xi);
    }


    auto solveJob = [&](u32 i) {
        if (auto error = solveOne(i, pdes[i].earlyExercise, t, x, v); !error.empty()) {
            std::cerr << "Fd1d::solve: " + error << std::endl;
        }
    };

    auto& pool = kw::ThreadPool::instance();
    for (auto i = 0; i < pdes.size(); i++) {
        pool.add_job(std::bind(solveJob, i));
    }
    pool.wait();

    return "";
}


Error
Fd1d::solveOne(const u64 ni, const bool earlyExercise, const Grid2d& t, const Grid2d& x, const Grid2d& v)
{
    auto tDim = t.rows();
    auto xDim = x.rows();

    for (int ti = tDim - 2; ti >= 0; --ti) {
        // Step 1
        //   - Calc B = [1 - θ dt 𝒜]
        //   - Calc W = [1 + (1 - θ) dt 𝒜] V(t + dt)

        f64 dt = t(ni, ti + 1) - t(ni, ti);
        {
            const auto xi = 0;

            const f64 inv_dx = 1. / (x(ni, xi + 1) - x(ni, xi));

            m_bl(ni, xi) = 0;
            m_b(ni, xi) = 1 - m_theta * dt * (m_a0(ni, ti) - inv_dx * m_ax(ni, ti));
            m_bu(ni, xi) = -m_theta * dt * (inv_dx * m_ax(ni, ti));

            m_w(ni, xi) = (1 + (1 - m_theta) * dt * m_a0(ni, ti)) * m_v(ni, xi) +
                (1 - m_theta) * dt * (m_ax(ni, ti) * inv_dx) * (m_v(ni, xi + 1) - m_v(ni, xi));
        }

        for (auto xi = 1; xi < xDim - 1; ++xi) {
            const f64 inv_dxu = 1. / (x(ni, xi + 1) - x(ni, xi));
            const f64 inv_dxm = 1. / (x(ni, xi + 1) - x(ni, xi - 1));
            const f64 inv_dxd = 1. / (x(ni, xi) - x(ni, xi - 1));

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
            const auto xi = xDim - 1;
            const f64 inv_dx = 1. / (x(ni, xi) - x(ni, xi - 1));

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
            &m_v(ni, 0)
        );

        // Step 3
        //  - Adjust for early exercise
        if (earlyExercise) {
            for (auto xi = 0; xi < xDim - 1; ++xi)
                m_v(ni, xi) = std::max(m_v(ni, xi), v(ni, xi));
        }
    }

    return "";
}


Error
Fd1d::value(const u64 ni, const f64 x_, const Grid2d& x, f64& v) const
{
    const auto xDim = x.rows();

    if (ni >= x.cols())
        return "Fd1d::value: Solution index out of range";

    u64 xi = 0;
    while ((xi < xDim) && (x(ni, xi) < x_))
        ++xi;

    if ((xi == 0) || (xi == xDim))
        return "Fd1d::value: x=" + std::to_string(x_) + " not in range (" + std::to_string(x(ni, 0))
            + ", " + std::to_string(x(ni, xDim - 1)) + ")";

    v = ((x(ni, xi) - x_) * m_v(ni, xi - 1) + (x_ - x(ni, xi - 1)) * m_v(ni, xi)) / (x(ni, xi) - x(ni, xi - 1));

    return "";
}
