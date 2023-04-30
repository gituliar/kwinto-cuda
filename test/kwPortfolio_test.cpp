#include <gtest/gtest.h>

#include <fstream>

#include "kwFd1d.h"
#include "kwPortfolio.h"



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
        m_config.tDim = 1024;
        m_config.xDim = 1024;

        const auto srcPath = "test/portfolio.csv";
        if (auto error = kw::loadPortfolio(srcPath, m_portfolio); !error.empty())
        {
            std::cerr << "kwPortfolioTest: Failed to open " << srcPath << '\n';
            return;
        }
    }

    kw::Fd1dConfig
        m_config;

    kw::Portfolio
        m_portfolio;
};

TEST_F(kwPortfolioTest, Fd1dCpu)
{
    ASSERT_EQ(m_portfolio.size(), 42000);
    std::vector<kw::Option> assets;
    for (const auto& [asset, _] : m_portfolio)
        assets.push_back(asset);

    kw::Fd1d<real> pricer;
    //kw::Fd1d_Gpu<real> pricer;

    m_config.pdeCount = assets.size();
    ASSERT_EQ(pricer.allocate(m_config), "");

    std::vector<kw::Fd1dPde<real>> pdes;
    ASSERT_EQ(kw::Fd1dPdeFor(assets, m_config, pdes), "");

    ASSERT_EQ(pricer.solve(pdes), "");

    size_t i = 0;
    for (const auto& [asset, want]: m_portfolio)
    {
        real got;
        if (auto error = pricer.value(i, asset.s, got); !error.empty())
        {
            std::cout << error << std::endl;
            continue;
        }
        EXPECT_NEAR(want, got, 0.04) << "spot = " << asset.s << "\nasset = " << asset << "\n";

        ++i;
    }

    ASSERT_EQ(pricer.free(), "");
}
