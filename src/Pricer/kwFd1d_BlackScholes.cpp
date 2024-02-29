#include "Pricer/kwF1d1_BlackScholes.h"

using namespace kw;


Error
Fd1d_BlackScholes_Pricer::init(const Config& config)
{
    if (auto err = m_fd1d.init(config); !err.empty())
        return "Fd1d_BlackScholes_Pricer::init : " + err;

    return "";
}

Error
Fd1d_BlackScholes_Pricer::price(const vector<Option>& assets, vector<f64>& prices)
{
    if (auto err = m_fd1d.price(assets, prices); !err.empty())
        return "Fd1d_BlackScholes_Pricer::price : " + err;

    auto assets_ = assets;
    for (auto& asset : assets_)
        asset.e = false;

    vector<f64> pricesFd1d;
    pricesFd1d.resize(prices.size());
    if (auto err = m_fd1d.price(assets_, pricesFd1d); !err.empty())
        return "Fd1d_BlackScholes_Pricer::price : " + err;

    vector<f64> pricesBs;
    pricesBs.resize(prices.size());

    BlackScholes_Pricer bs;
    if (auto err = m_bs.price(assets_, pricesBs); !err.empty())
        return "Fd1d_BlackScholes_Pricer::price : " + err;
    

    for (auto i = 0; i < prices.size(); i++) {
        prices[i] += pricesBs[i] - pricesFd1d[i];
    }

    return "";
}
