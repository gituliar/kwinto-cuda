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

    Fd1d<Real>::CpuGrid
            m_tGrid;
    Fd1d<Real>::CpuGrid
            m_vGrid; // Initial value condition (final payoff)
    Fd1d<Real>::CpuGrid
            m_xGrid;

public:
    Error   init(const Config& config)
    {
        const auto bCap = config.get("FD1D.BATCH_SIZE", 64);
        const auto tDim = config.get("FD1D.GRID_SIZE_T", 512);
        const auto xDim = config.get("FD1D.GRID_SIZE_X", 512);

        if (auto error = m_tGrid.init(bCap, tDim); !error.empty())
            return "Fd1dGpu_PriceEngine::init: " + error;

        if (auto error = m_xGrid.init(bCap, xDim); !error.empty())
            return "Fd1dGpu_PriceEngine::init: " + error;

        if (auto error = m_vGrid.init(bCap, xDim); !error.empty())
            return "Fd1dGpu_PriceEngine::init: " + error;

        if (auto error = m_solver.init(config, tDim, xDim); !error.empty())
            return "Fd1dGpu_PriceEngine::init: " + error;

        return "";
    }

    Error   price(size_t i, double spot, double& price) const override
    {
        Real price_;
        if (auto error = m_solver.value(i, spot, m_xGrid, price_); !error.empty())
            return "Fd1dGpu_PriceEngine::run: " + error;
        price = price_;

        return "";
    }

    Error   run(const std::vector<Option>& assets) override
    {
        if (auto error = Fd1dCpu_PriceEngine<Real>::initBatch(assets, m_batch, m_tGrid, m_xGrid, m_vGrid); !error.empty())
            return "Fd1dGpu_PriceEngine::run: " + error;

        if (auto error = m_solver.solve(m_batch, m_tGrid, m_xGrid, m_vGrid); !error.empty())
            return "Fd1dGpu_PriceEngine::run: " + error;

        return "";
    }
};

}