#pragma once

#include "Math/kwFd1d.h"
#include "kwPriceEngine.h"
#include "kwFd1dCpu.h"


namespace kw
{

template<typename Real>
class Fd1dGpu_PriceEngine : public PriceEngine
{
private:
    std::vector<Fd1dPde<Real>>
            m_batch;

    Fd1d_Gpu<Real>
            m_solver;

    Fd1d_Gpu<Real>::CpuGrid
            m_tGrid;
    Fd1d_Gpu<Real>::CpuGrid
            m_vGrid; // Initial value condition (final payoff)
    Fd1d_Gpu<Real>::CpuGrid
            m_xGrid;

public:
    Error
        init(const Config& config) override
    {
        const auto tDim = config.get("FD1D.T_GRID_SIZE", 512);
        const auto xDim = config.get("FD1D.X_GRID_SIZE", 512);

        if (auto error = m_solver.init(config, tDim, xDim); !error.empty())
            return "Fd1dGpu_PriceEngine::init: " + error;

        return "";
    }

    Error
        price(const std::vector<Option>& assets, std::vector<double>& prices) override
    {
        const auto n = assets.size();
        prices.resize(n);

        m_tGrid.resize(n, m_solver.tDim());
        m_xGrid.resize(n, m_solver.xDim());
        m_vGrid.resize(n, m_solver.xDim());

        if (auto error = Fd1dCpu_PriceEngine<Real>::initBatch(assets, m_batch, m_tGrid, m_xGrid, m_vGrid); !error.empty())
            return "Fd1dGpu_PriceEngine::price: " + error;

        if (auto error = m_solver.solve(m_batch, m_tGrid, m_xGrid, m_vGrid); !error.empty())
            return "Fd1dGpu_PriceEngine::price: " + error;


        for (auto i = 0; i < n; i++) {
            Real price_;
            if (auto error = m_solver.value(i, assets[i].s, m_xGrid, price_); !error.empty())
                return "Fd1dCpu_PriceEngine::price " + error;
            prices[i] = price_;
        }

        return "";
    }
};

}