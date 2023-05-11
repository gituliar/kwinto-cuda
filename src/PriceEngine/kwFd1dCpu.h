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
    Error   init(const Config& config)
    {
        const auto bCap = config.get("FD1D.BATCH_SIZ", 64);
        const auto tDim = config.get("FD1D.GRID_SIZE_T", 512);
        const auto xDim = config.get("FD1D.GRID_SIZE_X", 512);

        if (auto error = m_tGrid.init(bCap, tDim); !error.empty())
            return "Fd1dCpu_PriceEngine::init: " + error;

        if (auto error = m_xGrid.init(bCap, xDim); !error.empty())
            return "Fd1dCpu_PriceEngine::init: " + error;

        if (auto error = m_vGrid.init(bCap, xDim); !error.empty())
            return "Fd1dCpu_PriceEngine::init: " + error;

        if (auto error = m_solver.init(bCap, tDim, xDim); !error.empty())
            return "Fd1dCpu_PriceEngine::init: " + error;

        return "";
    }

    Error   price(size_t i, double spot, double& price) const override
    {
        Real price_;
        if (auto error = m_solver.value(i, spot, m_xGrid, price_); !error.empty())
            return "Fd1dCpu_PriceEngine::run: " + error;
        price = price_;

        return "";
    }

    Error   run(const std::vector<Option>& assets) override
    {
        if (auto error = initBatch(assets, m_batch, m_tGrid, m_xGrid, m_vGrid); !error.empty())
            return "Fd1dCpu_PriceEngine::run: " + error;

        if (auto error = m_solver.solve(m_batch, m_tGrid, m_xGrid, m_vGrid); !error.empty())
            return "Fd1dCpu_PriceEngine::run: " + error;

        return "";
    }

public:
    static Error
            initBatch(
                const std::vector<Option>& assets,
                std::vector<Fd1dPde<Real>>& batch,
                Fd1d<Real>::CpuGrid& tGrid,
                Fd1d<Real>::CpuGrid& xGrid,
                Fd1d<Real>::CpuGrid& vGrid)
    {
        // FIXME: Check if assets fit into grids
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

        const auto bCap = tGrid.cols();
        const auto tDim = tGrid.rows();
        const auto xDim = xGrid.rows();

        // Init T-Grid
        for (auto i = 0; i < batch.size(); ++i) {
            Real tMin = 0.0, tMax = batch[i].t;
            Real dt = (tMax - tMin) / (tDim - 1);

            for (auto j = 0; j < tDim; ++j)
                tGrid(i, j) = tMin + j * dt;
        }

        // Init V-Grid, X-Grid
        for (auto i = 0; i < assets.size(); ++i) {
            const auto& asset = assets[i];

            Real xMax = std::max<Real>(log(asset.k) + 10 * asset.z * sqrt(asset.t), log(5 * asset.k));
            Real xMin = std::min<Real>(log(asset.k) - 10 * asset.z * sqrt(asset.t), log(0.2 * asset.k));
            Real dx = (xMax - xMin) / (xDim - 1);

            for (auto j = 0; j < xDim; ++j) {
                Real xj = xMin + j * dx;

                xGrid(i, j) = xj;

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