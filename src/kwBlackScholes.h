#pragma once

#include "kwAsset.h"
#include "kwMath.h"

namespace kw
{

template<typename Real>
class BlackScholes
{
public:
    BlackScholes() :
        m_assets{}
    {};

    Error
        solve(const std::vector<Option<Real>>& assets)
    {
        m_assets = assets;

        return "";
    }

    Error
        value(const size_t ni, const Real s, Real& v) const
    {
        const auto& asset = m_assets[ni];

        const auto& k = asset.k;
        const auto& q = asset.q;
        const auto& r = asset.r;
        const auto& t = asset.t;
        const auto& z = asset.z;
        const auto  w = (asset.w == kParity::Put) ? -1 : 1;

        Real zt = z * std::sqrt(t);
        Real d1 = 1 / zt * (std::log(s / k) + (r - q + 0.5 * z * z) * t);
        Real d2 = d1 - zt;

        v = w * (s * cdfNormal(w * d1) * std::exp(-q * t) - k * cdfNormal(w * d2) * std::exp(-r * t));

        return "";
    }

private:
    std::vector<Option<Real>>
        m_assets;
};

}