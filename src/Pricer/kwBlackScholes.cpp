#include "Pricer/kwBlackScholes.h"

using namespace kw;


Error
BlackScholes_Pricer::init(const Config& config)
{
    return "";
}


Error
BlackScholes_Pricer::price(const std::vector<Option>& assets, std::vector<f64>& prices)
{
    prices.resize(assets.size());

    for (auto i = 0; i < assets.size(); i++) {
        if (auto err = priceOne(assets[i], prices[i]); !err.empty())
            return "BlackScholes_Pricer::price : " + err;
    }

    return "";
}


Error
BlackScholes_Pricer::priceOne(const Option& asset, f64& price) const
{
    if (asset.e) {
        price = nan;
        return "";
    }

    const f64 k = asset.k;
    const f64 q = asset.q;
    const f64 r = asset.r;
    const f64 s = asset.s;
    const f64 t = asset.t;
    const f64 z = asset.z;
    const i8  w = asset.w;

    f64 zt = z * std::sqrt(t);
    f64 d1 = 1 / zt * (std::log(s / k) + (r - q + 0.5 * z * z) * t);
    f64 d2 = d1 - zt;

    price = w * (s * cdfNormal(w * d1) * std::exp(-q * t) - k * cdfNormal(w * d2) * std::exp(-r * t));

    return "";
}