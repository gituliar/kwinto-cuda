#pragma once

#include "kwPriceEngine.h"
#include "Math/kwFd1d.h"


namespace kw
{

template<typename Real>
class Fd1dCpu_PriceEngine : public PriceEngine
{
private:
    std::vector<Fd1dPde<Real>>
        m_batch;

    Fd1d<Real>
        m_solver;

    Fd1d<Real>::CpuGrid
        m_tGrid;
    Fd1d<Real>::CpuGrid
        m_vGrid; // Initial value condition (final payoff)
    Fd1d<Real>::CpuGrid
        m_xGrid;

public:
    Error
        init(const Config& config) override
    {
        const auto tDim = config.get("FD1D.T_GRID_SIZE", 512);
        const auto xDim = config.get("FD1D.X_GRID_SIZE", 512);

        if (auto error = m_solver.init(tDim, xDim); !error.empty())
            return "Fd1dCpu_PriceEngine::init: " + error;

        return "";
    }

    Error
        price(size_t i, double spot, double& price) const override
    {
        Real price_;
        if (auto error = m_solver.value(i, spot, m_xGrid, price_); !error.empty())
            return "Fd1dCpu_PriceEngine::run: " + error;
        price = price_;

        return "";
    }

    Error
        run(const std::vector<Option>& assets) override
    {
        const auto n = assets.size();

        m_tGrid.resize(n, m_solver.tDim());
        m_xGrid.resize(n, m_solver.xDim());
        m_vGrid.resize(n, m_solver.xDim());

        if (auto error = initBatch(assets, m_batch, m_tGrid, m_xGrid, m_vGrid); !error.empty())
            return "Fd1dCpu_PriceEngine::run: " + error;

        if (auto error = m_solver.solve(m_batch, m_tGrid, m_xGrid, m_vGrid); !error.empty())
            return "Fd1dCpu_PriceEngine::run: " + error;

        return "";
    }

public:
    template<int Options>
    static Error
        initBatch(
            const std::vector<Option>& assets,
            std::vector<Fd1dPde<Real>>& batch,
            Vector2d<Real, Options>& tGrid,
            Vector2d<Real, Options>& xGrid,
            Vector2d<Real, Options>& vGrid)
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
        }

        const auto tDim = tGrid.rows();
        const auto xDim = xGrid.rows();

        // Init T-Grid
        for (auto i = 0; i < batch.size(); ++i) {
            Real tMin = 0.0, tMax = batch[i].t;
            Real dt = (tMax - tMin) / (tDim - 1);

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

            const Real density = 0.1;
            const Real scale = 10;

            const Real xMid = log(asset.s);
            const Real xMin = xMid - scale * asset.z * sqrt(asset.t);
            const Real xMax = xMid + scale * asset.z * sqrt(asset.t);

            const Real yMin = std::asinh((xMin - xMid) / density);
            const Real yMax = std::asinh((xMax - xMid) / density);

            const Real dy = 1. / (xDim - 1);
            for (auto j = 0; j < xDim; ++j) {
                const Real yj = j * dy;
                xGrid(i, j) = xMid + density * std::sinh(yMin * (1.0 - yj) + yMax * yj);
            }
        }

        // Init V-Grid
        for (auto i = 0; i < assets.size(); ++i) {
            const auto& asset = assets[i];
            for (auto j = 0; j < xDim; ++j) {
                const auto& xj = xGrid(i, j);
                if (asset.w == kParity::Put)
                    vGrid(i,j) = std::max<Real>(0, asset.k - std::exp(xj));
                else
                    vGrid(i,j) = std::max<Real>(0, std::exp(xj) - asset.k);
            }
        }

        return "";
    };
};

}