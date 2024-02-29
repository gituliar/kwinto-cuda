#pragma once

#include "Pricer/kwPricer.h"

#include "Math/kwMath.h"
#include "Math/kwFd1d.h"


namespace kw
{

///      𝒜 = -r + (r - z²/2) 𝒟x + z²/2 𝒟xx
///
class Fd1d_Pricer : public Pricer
{
private:
    f64 m_density;
    f64 m_scale;
    u64 m_tDim;
    u64 m_xDim;

    Fd1d
        m_solver;

    Grid2d
        m_t;
    Grid2d
        m_v; // Initial value condition (final payoff)
    Grid2d
        m_x;

public:
    Error
        init(const Config& config) final;
    Error
        price(const vector<Option>& assets, vector<f64>& prices) final;
};

}