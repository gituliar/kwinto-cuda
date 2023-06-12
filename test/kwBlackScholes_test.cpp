#include <gtest/gtest.h>

#include "Math/kwFd1d.h"
#include "kwBlackScholes.h"


using real = double;

class kwBlackScholesTest : public testing::Test {
protected:
    void
        SetUp() override
    {
        // For comparison see Section 77.13 in Quantitative Finance by Wilmott
        m_euroCall.k = static_cast<real>(100);
        m_euroCall.r = static_cast<real>(0.06);
        m_euroCall.q = static_cast<real>(0.02);
        m_euroCall.t = static_cast<real>(1.0);
        m_euroCall.z = static_cast<real>(0.2);
        m_euroCall.w = kw::Parity::Call;

        m_euroPut = m_euroCall;
        m_euroPut.w = kw::Parity::Put;
    }

    kw::Option
        m_euroPut;
    std::map<real, real>
        m_euroPutPrice = {
            {90., 10.627},
            {100., 5.885},
            {110., 2.987}
        };

    kw::Option
        m_euroCall;
    std::map<real, real>
        m_euroCallPrice = {
            {90., 4.668},
            {100., 9.729},
            {110., 16.633}
        };
};


TEST_F(kwBlackScholesTest, Exact)
{
    std::vector<kw::Option> assets;
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
