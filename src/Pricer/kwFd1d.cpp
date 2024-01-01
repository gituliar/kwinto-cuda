#include "Pricer/kwFd1d.h"

using namespace kw;


Error
Fd1d_Pricer::init(const Config& config)
{
    const auto tDim = config.get("FD1D.T_GRID_SIZE", 512);
    const auto xDim = config.get("FD1D.X_GRID_SIZE", 512);

    if (auto error = m_solver.init(tDim, xDim); !error.empty())
        return "Fd1d_Pricer::init: " + error;

    return "";
}

Error
Fd1d_Pricer::price(const std::vector<Option>& assets, std::vector<double>& prices)
{
    const auto n = assets.size();
    prices.resize(n);

    m_tGrid.resize(n, m_solver.tDim());
    m_xGrid.resize(n, m_solver.xDim());
    m_vGrid.resize(n, m_solver.xDim());

    if (auto error = initBatch(assets, m_batch, m_tGrid, m_xGrid, m_vGrid); !error.empty())
        return "Fd1d_Pricer::price: " + error;

    if (auto error = m_solver.solve(m_batch, m_tGrid, m_xGrid, m_vGrid); !error.empty())
        return "Fd1d_Pricer::price: " + error;


    for (auto i = 0; i < n; i++) {
        f64 price_;
        if (auto error = m_solver.value(i, assets[i].s, m_xGrid, price_); !error.empty())
            return "Fd1d_Pricer::price " + error;
        prices[i] = price_;
    }

    return "";
}

Error
Fd1d_Pricer::initBatch(
        const std::vector<Option>& assets,
        std::vector<Fd1dPde>& batch,
        Fd1d::CpuGrid& tGrid,
        Fd1d::CpuGrid& xGrid,
        Fd1d::CpuGrid& vGrid)
{
    batch.clear();
    batch.reserve(assets.size());

    for (const auto& a : assets)
    {
        auto& pde = batch.emplace_back();

        pde.t = a.t;

        pde.a0 = -a.r;
        pde.ax = a.r - a.q - a.z * a.z / 2;
        pde.axx = a.z * a.z / 2;
        
        pde.earlyExercise = a.e;
    }

    const auto tDim = tGrid.rows();
    const auto xDim = xGrid.rows();

    // Init T-Grid
    for (auto i = 0; i < batch.size(); ++i) {
        f64 tMin = 0.0, tMax = batch[i].t;
        f64 dt = (tMax - tMin) / (tDim - 1);

        for (auto j = 0; j < tDim; ++j)
            tGrid(i, j) = tMin + j * dt;
    }

    // Init X-Grid
    // 
    // https://github.com/lballabio/QuantLib/blob/master/ql/methods/finitedifferences/meshers/fdmblackscholesmesher.cpp
    // https://github.com/lballabio/QuantLib/blob/master/ql/pricingengines/vanilla/fdblackscholesvanillaengine.cpp
    // 
    for (auto i = 0; i < assets.size(); ++i) {
        const auto& asset = assets[i];

        const f64 density = 0.1;
        const f64 scale = 10;

        const f64 xMid = std::log(asset.s);
        const f64 xMin = xMid - scale * asset.z * std::sqrt(asset.t);
        const f64 xMax = xMid + scale * asset.z * std::sqrt(asset.t);

        const f64 yMin = std::asinh((xMin - xMid) / density);
        const f64 yMax = std::asinh((xMax - xMid) / density);

        const f64 dy = 1. / (xDim - 1);
        for (auto j = 0; j < xDim; ++j) {
            const f64 yj = j * dy;
            xGrid(i, j) = xMid + density * std::sinh(yMin * (1.0 - yj) + yMax * yj);
        }
    }

    // Init V-Grid
    for (auto i = 0; i < assets.size(); ++i) {
        const auto& asset = assets[i];
        for (auto j = 0; j < xDim; ++j) {
            const auto& xj = xGrid(i, j);
            if (asset.w == Parity::Put)
                vGrid(i,j) = std::max<f64>(0, asset.k - std::exp(xj));
            else
                vGrid(i,j) = std::max<f64>(0, std::exp(xj) - asset.k);
        }
    }

    return "";
};
