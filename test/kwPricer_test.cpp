#include <gtest/gtest.h>

#include "Core/kwAsset.h"
#include "Core/kwConfig.h"
#include "Pricer/kwPricerFactory.h"

using namespace kw;


class kwPricerTest : public testing::Test {
protected:
    void
        SetUp() override
    {
        m_testData = {
            {{1.0, 100., 0.2, 0.06, 0.02,  90., 0, -1}, 10.627},
            {{1.0, 100., 0.2, 0.06, 0.02, 100., 0, -1}, 5.885},
            {{1.0, 100., 0.2, 0.06, 0.02, 110., 0, -1}, 2.987},
            {{1.0, 100., 0.2, 0.06, 0.02,  90., 0, +1}, 4.668},
            {{1.0, 100., 0.2, 0.06, 0.02, 100., 0, +1}, 9.729},
            {{1.0, 100., 0.2, 0.06, 0.02, 110., 0, +1}, 16.633},

            {{1.0, 100., 0.2, 0.06, 0.08,  90., 1, -1}, 13.988121682},
            {{1.0, 100., 0.2, 0.06, 0.08, 100., 1, -1}, 8.409190396},
            {{1.0, 100., 0.2, 0.06, 0.08, 110., 1, -1}, 4.6592955111},
            {{1.0, 100., 0.2, 0.06, 0.08,  90., 1, +1}, 2.9472036256},
            {{1.0, 100., 0.2, 0.06, 0.08, 100., 1, +1}, 6.8422540642},
            {{1.0, 100., 0.2, 0.06, 0.08, 110., 1, +1}, 12.7940107108}
        };
    }

    std::vector<std::pair<Option, f64>>
        m_testData;
};


TEST_F(kwPricerTest, BlackScholes)
{
    std::vector<Option> assets;
    for (const auto& test : m_testData)
        assets.push_back(test.first);

    Config config;
    config.set("PRICER", "BS");

    kw::sPtr<Pricer> pricer;
    ASSERT_EQ(PricerFactory::create(config, pricer), "");

    std::vector<f64> prices;
    ASSERT_EQ(pricer->price(assets, prices), "");

    for (auto i = 0; i < m_testData.size(); ++i) {
        const auto& want = m_testData[i].second;
        const auto& got = prices[i];
        if (isnan(got))
            continue;

        EXPECT_NEAR(want, got, 1.3e-3);
    }
}


TEST_F(kwPricerTest, Fd1d)
{
    std::vector<Option> assets;
    for (const auto& test : m_testData)
        assets.push_back(test.first);

    kw::Config config;
    config.set("PRICER", "FD1D");

    kw::sPtr<kw::Pricer> pricer;
    ASSERT_EQ(kw::PricerFactory::create(config, pricer), "");

    std::vector<f64> prices;
    ASSERT_EQ(pricer->price(assets, prices), "");

    for (auto i = 0; i < m_testData.size(); ++i) {
        const auto& want = m_testData[i].second;
        const auto& got = prices[i];

        EXPECT_NEAR(want, got, 1.3e-3);
    }
}


TEST_F(kwPricerTest, Fd1dBlackScholes)
{
    std::vector<Option> assets;
    for (const auto& test : m_testData)
        assets.push_back(test.first);

    kw::Config config;
    config.set("PRICER", "FD1D-BS");

    kw::sPtr<kw::Pricer> pricer;
    ASSERT_EQ(kw::PricerFactory::create(config, pricer), "");

    std::vector<f64> prices;
    ASSERT_EQ(pricer->price(assets, prices), "");

    for (auto i = 0; i < m_testData.size(); ++i) {
        const auto& want = m_testData[i].second;
        const auto& got = prices[i];

        EXPECT_NEAR(want, got, 1.3e-3);
    }
}
