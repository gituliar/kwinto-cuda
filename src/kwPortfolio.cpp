#include "kwPortfolio.h"

#include <filesystem>
#include <fstream>

#include "kwString.h"


kw::Error
kw::loadPortfolio(const std::string& srcPath_, Portfolio& portfolio)
{
    const auto srcPath = std::filesystem::absolute(srcPath_);
    std::ifstream src(srcPath);
    if (!src.is_open())
        return "loadPortfolio: Failed to open " + srcPath.string();

    int e, k, q, r, s, t, v, w, z;
    {
        std::string header;
        std::getline(src, header);

        int i = 0;
        e = k = q = r = s = t = v = w = z = -1;
        for (const auto& colName : kw::split(header, ','))
        {
            if (colName == "early_exercise")
                e = i;
            else if (colName == "strike")
                k = i;
            else if (colName == "dividend_rate")
                q = i;
            else if (colName == "interest_rate")
                r = i;
            else if (colName == "spot")
                s = i;
            else if (colName == "time_to_maturity")
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
            error << "loadPortfolio: Some option data is missing: e=" << e << ", k=" << k << ", q=" << q
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
        asset.e = vals[e] == "a";
        asset.w = vals[w] == "c" ? kw::kParity::Call : kw::kParity::Put;

        double price;
        kw::fromString(vals[v], price);

        portfolio[asset] = price;
    }

    return "";
}
