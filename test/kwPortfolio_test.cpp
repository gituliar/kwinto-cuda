#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>

#include "kwFd1d.h"
#include "kwString.h"



std::vector<std::string> split(const std::string& src, char delim)
{
    std::vector<std::string> result;
    std::stringstream buf(src);

    for (std::string item; getline(buf, item, delim);) {
        result.push_back(item);
    }

    return result;
}


using real = double;

class kwPortfolioTest : public testing::Test {
protected:
    void
        SetUp() override
    {
        m_config.theta = 0.5;
        m_config.tDim = 2048;
        m_config.xDim = 2048;

        std::filesystem::path srcPath = "C:\\Users\\Sasha\\Sources\\kwinto-cuda\\portfolio.csv";
        std::ifstream src(srcPath);
        if (!src.is_open())
        {
            std::cout << "kwPortfolioTest: Failed to open " << srcPath << '\n';
            return;
        }

        {
            std::string header;
            std::getline(src, header);

            int i = 0, e, q, r, s, t, v, w, z;
            e = q = r = s = t = v = w = z = -1;
            for (const auto& colName : split(header, ','))
            {
                if (colName == "early_exercise")
                    e = i;
                else if (colName == "dividend_rate")
                    q = i;
                else if (colName == "interest_rate")
                    r = i;
                else if (colName == "spot")
                    s = i;
                else if (colName == "t")
                    t = i;
                else if (colName == "price")
                    v = i;
                else if (colName == "parity")
                    w = i;
                else if (colName == "volatility")
                    z = i;

                ++i;
            }
            if (e == -1 || q == -1 || r == -1 || s == -1 || t == -1 || v == -1 || w == -1 || z == -1)
            {
                std::cout << "kwPortfolioTest: Some option data is missing: e=" << e << ", q=" << q << ", r=" << r
                    << ", s=" << s << ", t=" << t << ", v=" << v << ", w=" << w << ", z=" << z << std::endl;
                return;
            }

            std::vector<std::string> vals;
            for (std::string line; std::getline(src, line);)
            {
                vals = split(line, ',');

                real price, spot;
                kw::fromString(vals[v], price);
                kw::fromString(vals[s], spot);

                kw::Option<real> asset;
                asset.k = 100;
                kw::fromString(vals[t], asset.t);
                kw::fromString(vals[z], asset.z);
                kw::fromString(vals[q], asset.q);
                kw::fromString(vals[r], asset.r);
                asset.e = (vals[e] == "a");
                asset.w = vals[w] == "c" ? kw::kParity::Call : kw::kParity::Put;

                //if (asset.t != 1.0 || asset.z != 0.2 || asset.r != 0.06 || asset.q != 0.08 || spot != 100)
                //if (asset.t != 1.0)
                //    continue;

                m_testData[asset].emplace_back(spot, price);
            }

        }
    }

    kw::Fd1dConfig
        m_config;

    std::map<kw::Option<real>, std::vector<std::pair<real, real>>>
        m_testData;
};

TEST_F(kwPortfolioTest, Fd1d)
{
    ASSERT_EQ(m_testData.size(), 4200);
    std::vector<kw::Option<real>> assets;
    for (const auto& [asset, tests] : m_testData)
        assets.push_back(asset);

    kw::Fd1d<real> pricer;
    //kw::Fd1d_Gpu<real> pricer;

    m_config.pdeCount = assets.size();
    ASSERT_EQ(pricer.allocate(m_config), "");

    std::vector<kw::Fd1dPde<real>> pdes;
    ASSERT_EQ(kw::Fd1dPdeFor(assets, m_config, pdes), "");

    ASSERT_EQ(pricer.solve(pdes), "");

    size_t i = 0;
    for (const auto& [asset, tests]: m_testData)
    {
        for (const auto& [spot, want] : tests)
        {
            real got;
            if (auto error = pricer.value(i, spot, got); !error.empty())
            {
                std::cout << error << std::endl;
                continue;
            }
            EXPECT_NEAR(want, got, 0.02) << "spot = " << spot << "\nasset = " << asset << "\n";
        }

        ++i;
    }

    ASSERT_EQ(pricer.free(), "");
}
