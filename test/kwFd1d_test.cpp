#include <gtest/gtest.h>

#include "Math/kwFd1d.h"
#include "PriceEngine/kwPriceEngineFactory.h"


using real = double;

class kwFd1dTest : public testing::Test {
protected:
    void
        SetUp() override
    {
        //m_testData = {
        //    {{1.0, 100., 0.2, 0.06, 0.08, 100., true, kw::Parity::Put}, {
        //        {90, 13.988}, {100, 8.409}, {110, 4.659}}},
        //    {{1.0, 100., 0.2, 0.06, 0.08, 100., true, kw::Parity::Call}, {
        //        {90, 2.947}, {100, 6.842}, {110, 12.794}}},
        //};
        m_testData = {
            {{1.0, 100., 0.2, 0.06, 0.08, 90., true, kw::Parity::Put}, 13.988},
            {{1.0, 100., 0.2, 0.06, 0.08, 100., true, kw::Parity::Put}, 8.409},
            {{1.0, 100., 0.2, 0.06, 0.08, 110., true, kw::Parity::Put}, 4.659},
            {{1.0, 100., 0.2, 0.06, 0.08, 90., true, kw::Parity::Call}, 2.947},
            {{1.0, 100., 0.2, 0.06, 0.08, 100., true, kw::Parity::Call}, 6.842},
            {{1.0, 100., 0.2, 0.06, 0.08, 110., true, kw::Parity::Call}, 12.794}
        };

        m_config.set("FD1D.T_GRID_SIZE", 512);
        m_config.set("FD1D.X_GRID_SIZE", 512);
    }

    //std::vector<std::pair<kw::Option, std::vector<std::pair<real, real>>>>
    //    m_testData;

    std::vector<std::pair<kw::Option, real>>
        m_testData;

    kw::Config
        m_config;
};


TEST_F(kwFd1dTest, Fd1dCpu)
{
    std::vector<kw::Option> assets;
    for (const auto& test : m_testData)
        assets.push_back(test.first);

    kw::sPtr<kw::PriceEngine> engine;
    m_config.set("PRICE_ENGINE.MODE", "FD1D_CPU64");
    ASSERT_EQ(kw::PriceEngineFactory::create(m_config, engine), "");

    std::vector<double> prices;
    ASSERT_EQ(engine->price(assets, prices), "");

    for (auto i = 0; i < m_testData.size(); ++i) {
        const auto& want = m_testData[i].second;
        const auto& got = prices[i];
        EXPECT_NEAR(want, got, 1.3e-3);
    }
}


TEST_F(kwFd1dTest, Fd1dGpu)
{
    std::vector<kw::Option> assets;
    for (const auto& test : m_testData)
        assets.push_back(test.first);

    kw::sPtr<kw::PriceEngine> engine;
    m_config.set("PRICE_ENGINE.MODE", "FD1D_GPU64");
    ASSERT_EQ(kw::PriceEngineFactory::create(m_config, engine), "");

    std::vector<double> prices;
    ASSERT_EQ(engine->price(assets, prices), "");

    for (auto i = 0; i < m_testData.size(); ++i) {
        const auto& want = m_testData[i].second;
        const auto& got = prices[i];
        EXPECT_NEAR(want, got, 1.3e-3);
    }
}


TEST_F(kwFd1dTest, Fd1dQl)
{
    std::vector<kw::Option> assets;
    for (const auto& test : m_testData)
        assets.push_back(test.first);

    kw::sPtr<kw::PriceEngine> engine;
    m_config.set("PRICE_ENGINE.MODE", "FD1D_QL");
    ASSERT_EQ(kw::PriceEngineFactory::create(m_config, engine), "");

    std::vector<double> prices;
    ASSERT_EQ(engine->price(assets, prices), "");

    for (auto i = 0; i < m_testData.size(); ++i) {
        const auto& want = m_testData[i].second;
        const auto& got = prices[i];
        EXPECT_NEAR(want, got, 1.3e-3);
    }
}
