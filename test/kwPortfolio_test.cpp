#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>

#include "kwPortfolio.h"
#include "PriceEngine/kwPriceEngineFactory.h"



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
        m_config.set("FD1D.THETA", 0.5);
        m_config.set("FD1D.T_GRID_DIM", 1024);
        m_config.set("FD1D.X_GRID_DIM", 1024);

        const auto srcPath = "portfolio.csv";
        if (auto error = kw::loadPortfolio(srcPath, m_portfolio); !error.empty())
        {
            std::cerr << "kwPortfolioTest: " << error << '\n';
            return;
        }
    }

    kw::Config
        m_config;

    kw::Portfolio
        m_portfolio;
};

TEST_F(kwPortfolioTest, Fd1dCpu)
{
    ASSERT_EQ(m_portfolio.size(), 24000);

    std::vector<kw::Option> assets;
    for (const auto& [asset, _] : m_portfolio) {
        if (!asset.e)
            continue;

        assets.push_back(asset);
    }

    kw::Config config;
    config.set("PRICE_ENGINE.MODE", "FD1D_CPU64");

    kw::sPtr<kw::PriceEngine> engine;
    ASSERT_EQ(kw::PriceEngineFactory::create(config, engine), "");

    std::vector<double> prices;
    ASSERT_EQ(engine->price(assets, prices), "");

    for (int i = 0; i< assets.size(); i++)
    {
        const auto& asset = assets[i];

        const auto& want = m_portfolio[asset];
        const auto& got = prices[i];

        EXPECT_NEAR(want, got, 0.04) << "spot = " << asset.s << "\nasset = " << asset << "\n";

        ++i;
    }
}
