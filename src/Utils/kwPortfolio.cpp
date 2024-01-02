#include "kwPortfolio.h"

#include <filesystem>
#include <fstream>
#include <iostream>

#include "Core/kwString.h"
#include "Pricer/kwPricerFactory.h"

using namespace kw;


Error
Portfolio::load(const std::string& srcPath_)
{
    const auto srcPath = std::filesystem::absolute(srcPath_);
    std::ifstream src(srcPath);
    if (!src.is_open())
        return "Portfolio::load : Failed to open " + srcPath.string();

    int e, k, q, r, s, t, v, w, z;
    {
        std::string header;
        std::getline(src, header);

        int i = 0;
        e = k = q = r = s = t = v = w = z = -1;
        for (const auto& colName : kw::split(header, ','))
        {
            if (colName == "exercise")
                e = i;
            else if (colName == "strike")
                k = i;
            else if (colName == "dividend_rate")
                q = i;
            else if (colName == "interest_rate")
                r = i;
            else if (colName == "spot")
                s = i;
            else if (colName == "expiry")
                t = i;
            else if (colName == "price")
                v = i;
            else if (colName == "parity")
                w = i;
            else if (colName == "volatility")
                z = i;

            ++i;
        }
        if (e == -1 || k == -1 || q == -1 || r == -1 || s == -1 || t == -1 || v == -1 || w == -1 || z == -1)
        {
            std::stringstream error;
            error << "Portfolio::load : Some option data is missing: e=" << e << ", k=" << k << ", q=" << q
                << ", r=" << r << ", s=" << s << ", t=" << t << ", v=" << v << ", w=" << w << ", z=" << z;
            return error.str();
        }
    }

    for (std::string line; std::getline(src, line);)
    {
        auto vals = kw::split(line, ',');

        kw::Option asset;
        kw::fromString(vals[k], asset.k);
        kw::fromString(vals[q], asset.q);
        kw::fromString(vals[r], asset.r);
        kw::fromString(vals[t], asset.t);
        kw::fromString(vals[s], asset.s);
        kw::fromString(vals[z], asset.z);
        asset.e = vals[e] == "a" ? 1 : 0;
        asset.w = vals[w] == "c" ? +1 : -1;

        m_assets.push_back(asset);


        f64 price;
        kw::fromString(vals[v], price);

        m_prices.push_back(price);
    }

    return "";
}


Error
Portfolio::price(const Config& config, std::vector<f64>& prices)
{
    sPtr<Pricer> pricer;
    if (auto err = PricerFactory::create(config, pricer); !err.empty())
        return "Portfolio::price : " + err;

    if (auto err = pricer->price(m_assets, prices); !err.empty())
        return "Portfolio::price : " + err;

    return "";
}


Error
Portfolio::printPricesStats(const std::vector<f64>& prices)
{
    f64 absDiffSum1 = 0, absDiffSum2 = 0, relDiffSum1 = 0, relDiffSum2 = 0;
    u64 absDiffSize = 0, relDiffSize = 0;
    f64 mae = 0, mre = 0; // Maximum Absolute / Relative Error

    Option maeAsset, mreAsset;

    f64 tolerance = 0.5;

    for (auto j = 0; j < m_prices.size(); j++) {
        const auto& asset = m_assets[j];
        const auto& wantPrice = m_prices[j];

        const auto gotPrice = prices[j];

        if (wantPrice < tolerance)
            continue;

        double absDiff = std::abs(wantPrice - gotPrice);
        double relDiff = absDiff / wantPrice;

        if (absDiff > mae) {
            mae = absDiff;
            maeAsset = asset;
        }
        if (relDiff > mre) {
            mre = relDiff;
            mreAsset = asset;
        }

        absDiffSum1 += absDiff;
        absDiffSum2 += absDiff * absDiff;
        absDiffSize++;

        relDiffSum1 += relDiff;
        relDiffSum2 += relDiff * relDiff;
        relDiffSize++;
    }

    {
        auto absDiffMean = absDiffSum1 / absDiffSize;
        auto rmse = std::sqrt(absDiffSum2 / absDiffSize - absDiffMean * absDiffMean);

        auto relDiffMean = relDiffSum1 / relDiffSize;
        auto rrmse = std::sqrt(relDiffSum2 / relDiffSize - relDiffMean * relDiffMean);

        std::cout << "Price Statistics\n";
        std::cout << std::scientific;
        std::cout << "       RMSE : " << rmse << std::endl;
        std::cout << "      RRMSE : " << rrmse << std::endl;
        std::cout << "        MAE : " << mae << std::endl;
        std::cout << "        MRE : " << mre << std::endl;
        std::cout << "  MAE Asset : " << maeAsset.asString() << std::endl;
        std::cout << "  MRE Asset : " << mreAsset.asString() << std::endl;
        std::cout << std::fixed;
        std::cout << "      total : " << absDiffSize << " options" << std::endl;
        std::cout << std::endl;
    }

    return "";
}
