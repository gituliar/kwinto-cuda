#include "Pricer/kwFd1d.h"

using namespace kw;


Error
Fd1d_Pricer::init(const Config& config)
{
    m_tDim = config.get("FD1D.T_GRID_SIZE", 512);
    m_xDim = config.get("FD1D.X_GRID_SIZE", 512);

    return "";
}

Error
Fd1d_Pricer::price(const vector<Option>& assets, vector<f64>& prices)
{
    /// TODO: Compress options with from the same chain

    /// Allocate working memory
    ///
    const auto n = assets.size();

    m_t.resize(n, m_tDim);
    m_x.resize(n, m_xDim);
    m_v.resize(n, m_xDim);

    vector<Fd1dPde> pdes;
    pdes.reserve(n);

    /// Fill PDE coefficients
    ///
    for (const auto& asset : assets) {
        auto& pde = pdes.emplace_back();

        pde.t = asset.t;

        pde.a0 = -asset.r;
        pde.ax = asset.r - asset.q - asset.z * asset.z / 2;
        pde.axx = asset.z * asset.z / 2;
        
        pde.earlyExercise = asset.e;
    }

    /// Fill t-grid
    ///
    for (auto i = 0; i < pdes.size(); ++i) {
        f64 tMin = 0.0, tMax = pdes[i].t;
        f64 dt = (tMax - tMin) / (m_tDim - 1);

        for (auto j = 0; j < m_tDim; ++j)
            m_t(i, j) = tMin + j * dt;
    }

    /// Fill x-grid
    /// 
    /// https://github.com/lballabio/QuantLib/blob/master/ql/methods/finitedifferences/meshers/fdmblackscholesmesher.cpp
    /// https://github.com/lballabio/QuantLib/blob/master/ql/pricingengines/vanilla/fdblackscholesvanillaengine.cpp
    /// 
    for (auto i = 0; i < assets.size(); ++i) {
        const auto& asset = assets[i];

        const f64 density = 0.5;
        const f64 scale = 50;

        const f64 xMid = 0.; // log(asset.s / asset.k);
        const f64 xMin = xMid - scale * asset.z * sqrt(asset.t);
        const f64 xMax = xMid + scale * asset.z * sqrt(asset.t);

        const f64 yMin = asinh((xMin - xMid) / density);
        const f64 yMax = asinh((xMax - xMid) / density);

        const f64 dy = 1. / (m_xDim - 1);
        for (auto j = 0; j < m_xDim; ++j) {
            const f64 yj = j * dy;
            m_x(i, j) = xMid + density * sinh(yMin * (1.0 - yj) + yMax * yj);
        }
    }

    /// Fill v-grid
    ///
    for (auto i = 0; i < assets.size(); ++i) {
        for (auto j = 0; j < m_xDim; ++j) {
            if (assets[i].w < 0)
                /// put
                m_v(i,j) = std::max<f64>(0, 1. - exp(m_x(i,j)));
            else
                /// call
                m_v(i,j) = std::max<f64>(0, exp(m_x(i,j)) - 1.);
        }
    }

    /// Solve
    ///
    if (auto error = m_solver.solve(pdes, m_t, m_x, m_v); !error.empty())
        return "Fd1d_Pricer::price: " + error;

    /// Fill prices
    ///
    prices.resize(n);
    for (auto i = 0; i < n; i++) {
        const auto& asset = assets[i];

        f64 price_;
        auto x = log(asset.s / asset.k);
        if (auto err = m_solver.value(i, x, m_x, price_); !err.empty())
            return "Fd1d_Pricer::price " + err;
        prices[i] = asset.k * price_;
    }

    return "";
}
