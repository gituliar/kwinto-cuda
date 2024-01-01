#include <gtest/gtest.h>

#include "Pricer/kwBlackScholes.h"

using namespace kw;


class kwBlackScholesTest : public testing::Test {
protected:
    void
        SetUp() override
    {
        m_testData = {
            {{1.0, 100., 0.2, 0.06, 0.02, 90., true, Parity::Put}, 10.627},
            {{1.0, 100., 0.2, 0.06, 0.02, 100., true, Parity::Put}, 5.885},
            {{1.0, 100., 0.2, 0.06, 0.02, 110., true, Parity::Put}, 2.987},
            {{1.0, 100., 0.2, 0.06, 0.02, 90., true, Parity::Call}, 4.668},
            {{1.0, 100., 0.2, 0.06, 0.02, 100., true, Parity::Call}, 9.729},
            {{1.0, 100., 0.2, 0.06, 0.02, 110., true, Parity::Call}, 16.633}
        };
    }

    std::vector<std::pair<Option, f64>>
        m_testData;
};


TEST_F(kwBlackScholesTest, Exact)
{
    std::vector<Option> assets;
    for (const auto& test : m_testData)
        assets.push_back(test.first);

    BlackScholes_Pricer pricer;

    std::vector<f64> prices;
    ASSERT_EQ(pricer.price(assets, prices), "");

    for (auto i = 0; i < m_testData.size(); ++i) {
        const auto& want = m_testData[i].second;
        const auto& got = prices[i];
        EXPECT_NEAR(want, got, 1.3e-3);
    }
}
