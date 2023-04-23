#include <gtest/gtest.h>

#include "kwBlackScholes.h"
#include "kwFd1d.h"


using real = double;

class kwEuropeanTest : public testing::Test {
protected:
    void
        SetUp() override
    {
        m_config.theta = 0.5;
        m_config.tDim = 768;
        m_config.xDim = 1024;

        // For comparison see Section 77.13 in Quantitative Finance by Wilmott
        m_euroCall.e = false;
        m_euroCall.k = static_cast<real>(100);
        m_euroCall.r = static_cast<real>(0.06);
        m_euroCall.q = static_cast<real>(0.02);
        m_euroCall.t = static_cast<real>(1.0);
        m_euroCall.z = static_cast<real>(0.2);
        m_euroCall.w = kw::kParity::Call;

        m_euroPut = m_euroCall;
        m_euroPut.w = kw::kParity::Put;
    }

    kw::Option<real>
        m_euroPut;
    std::map<real, real>
        m_euroPutPrice = {
            {90., 10.627},
            {100., 5.885},
            {110., 2.987}
        };

    kw::Option<real>
        m_euroCall;
    std::map<real, real>
        m_euroCallPrice = {
            {90., 4.668},
            {100., 9.729},
            {110., 16.633}
        };

    kw::Fd1dConfig
        m_config;
};


TEST_F(kwEuropeanTest, BlackScholes)
{
    std::vector<kw::Option<real>> assets;
    assets.push_back(m_euroPut);
    assets.push_back(m_euroCall);

    kw::BlackScholes<real> pricer;

    pricer.solve(assets);


    // Put
    for (const auto& [s, price] : m_euroPutPrice)
    {
        real v;
        ASSERT_EQ(pricer.value(0, s, v), "");
        EXPECT_NEAR(v, price, 1e-2) << " s = " << s;
    }

    // Call
    for (const auto& [s, price] : m_euroCallPrice)
    {
        real v;
        ASSERT_EQ(pricer.value(1, s, v), "");
        EXPECT_NEAR(v, price, 1e-2);
    }
}


TEST_F(kwEuropeanTest, Fd1d)
{
    std::vector<kw::Option<real>> assets;
    assets.push_back(m_euroPut);
    assets.push_back(m_euroCall);

    kw::Fd1d<real> pricer;

    m_config.pdeCount = assets.size();
    ASSERT_EQ(pricer.allocate(m_config), "");

    std::vector<kw::Fd1dPde<real>> pdes;
    ASSERT_EQ(kw::Fd1dPdeFor(assets, m_config, pdes), "");

    ASSERT_EQ(pricer.solve(pdes), "");

    // Put
    for (const auto& [s, price] : m_euroPutPrice)
    {
        real v;
        ASSERT_EQ(pricer.value(0, s, v), "");
        EXPECT_NEAR(v, price, 1e-2);
    }

    // Call
    for (const auto& [s, price] : m_euroCallPrice)
    {
        real v;
        ASSERT_EQ(pricer.value(1, s, v), "");
        EXPECT_NEAR(v, price, 1e-2);
    }

    ASSERT_EQ(pricer.free(), "");
}


TEST_F(kwEuropeanTest, Fd1d_GPU)
{
    std::vector<kw::Option<real>> assets;
    assets.push_back(m_euroPut);
    assets.push_back(m_euroCall);

    kw::Fd1d_Gpu<real> pricer;

    m_config.pdeCount = assets.size();
    ASSERT_EQ(pricer.allocate(m_config), "");

    std::vector<kw::Fd1dPde<real>> pdes;
    ASSERT_EQ(kw::Fd1dPdeFor(assets, m_config, pdes), "");

    ASSERT_EQ(pricer.solve(pdes), "");

    // Put
    for (const auto& [s, price] : m_euroPutPrice)
    {
        real v;
        ASSERT_EQ(pricer.value(0, s, v), "");
        EXPECT_NEAR(v, price, 1e-2);
    }

    // Call
    for (const auto& [s, price] : m_euroCallPrice)
    {
        real v;
        ASSERT_EQ(pricer.value(1, s, v), "");
        EXPECT_NEAR(v, price, 1e-2);
    }

    ASSERT_EQ(pricer.free(), "");
}
