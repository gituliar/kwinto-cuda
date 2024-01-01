#pragma once

#include "kwAsset.h"
#include "Math/kwMath.h"

namespace kw
{

class BlackScholes
{
public:
    BlackScholes() :
        m_assets{}
    {};

    Error
        solve(const std::vector<Option>& assets)
    {
        m_assets = assets;

        return "";
    }

    Error
        value(const size_t ni, const f64 s, f64& v) const
    {
        const auto& asset = m_assets[ni];

        const f64 k = asset.k;
        const f64 q = asset.q;
        const f64 r = asset.r;
        const f64 t = asset.t;
        const f64 z = asset.z;
        const auto w = (asset.w == Parity::Put) ? -1 : 1;

        f64 zt = z * std::sqrt(t);
        f64 d1 = 1 / zt * (std::log(s / k) + (r - q + 0.5 * z * z) * t);
        f64 d2 = d1 - zt;

        v = w * (s * cdfNormal(w * d1) * std::exp(-q * t) - k * cdfNormal(w * d2) * std::exp(-r * t));

        return "";
    }

private:
    std::vector<Option>
        m_assets;
};

}