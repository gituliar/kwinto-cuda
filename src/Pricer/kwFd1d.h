#pragma once

#include "Pricer/kwPricer.h"

#include "Math/kwMath.h"
#include "Math/kwFd1d.h"


namespace kw
{

class Fd1d_Pricer : public Pricer
{
private:
    std::vector<Fd1dPde>
        m_batch;

    Fd1d
        m_solver;

    Fd1d::CpuGrid
        m_tGrid;
    Fd1d::CpuGrid
        m_vGrid; // Initial value condition (final payoff)
    Fd1d::CpuGrid
        m_xGrid;

public:
    Error
        init(const Config& config) final;

    Error
        price(const std::vector<Option>& assets, std::vector<f64>& prices) final;

public:
    static Error
        initBatch(
            const std::vector<Option>& assets,
            std::vector<Fd1dPde>& batch,
            Fd1d::CpuGrid& tGrid,
            Fd1d::CpuGrid& xGrid,
            Fd1d::CpuGrid& vGrid);
};

}