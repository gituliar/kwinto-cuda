#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>

#include "Pricer/kwPricerFactory.h"
#include "Utils/kwPortfolio.h"

using namespace kw;



class kwPortfolioTest : public testing::Test {
protected:
    void
        SetUp() override
    {

        if (auto err = m_portfolio.load("test/portfolio_qdfp.csv"); !err.empty()) {
            std::cerr << "kwPortfolioTest: " << err << '\n';
            return;
        }
    }

    kw::Portfolio
        m_portfolio;
};

TEST_F(kwPortfolioTest, Fd1d)
{
    ASSERT_EQ(m_portfolio.assets().size(), 6000);

    std::vector<kw::Option> assets;
    for (const auto& asset : m_portfolio.assets()) {
        if (!asset.e)
            continue;

        assets.push_back(asset);
    }

    kw::Config config;
    config.set("PRICER", "FD1D");
    config.set("FD1D.THETA", 0.5);
    config.set("FD1D.T_GRID_DIM", 1024);
    config.set("FD1D.X_GRID_DIM", 1024);

    kw::sPtr<kw::Pricer> engine;
    ASSERT_EQ(kw::PricerFactory::create(config, engine), "");

    std::vector<double> prices;
    ASSERT_EQ(engine->price(assets, prices), "");

    for (int i = 0; i< assets.size(); i++) {
        const auto wantPrice = m_portfolio.prices()[i];
        const auto gotPrice = prices[i];

        EXPECT_NEAR(wantPrice, gotPrice, 0.005) << "spot = " << assets[i].s << "\nasset = " << assets[i] << "\n";
    }
}
