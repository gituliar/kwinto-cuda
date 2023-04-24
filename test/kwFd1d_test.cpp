#include <gtest/gtest.h>

#include "kwFd1d.h"


using real = double;

class kwFd1dTest : public testing::Test {
protected:
    void
        SetUp() override
    {
        m_config.theta = 0.5;
        m_config.tDim = 768;
        m_config.xDim = 1024;

        m_testData = {
            {{1.0, 100., 0.2, 0.06, 0.08, true, kw::kParity::Put}, {
                {90, 13.988}, {100, 8.409}, {110, 4.659}}},
            {{1.0, 100., 0.2, 0.06, 0.08, true, kw::kParity::Call}, {
                {90, 2.947}, {100, 6.842}, {110, 12.794}}},
            {{1.0, 100., 0.2, 0.06, 0.08, false, kw::kParity::Put}, {
                {90, 13.944}, {100, 8.397}, {110, 4.656}}},
            {{1.0, 100., 0.2, 0.06, 0.08, false, kw::kParity::Call}, {
                {90, 2.848}, {100, 6.532}, {110, 12.022}}},
        };
    }

    std::vector<std::pair<kw::Option<real>, std::vector<std::pair<real, real>>>>
        m_testData;

    kw::Fd1dConfig
        m_config;
};


TEST_F(kwFd1dTest, Cpu)
{
    std::vector<kw::Option<real>> assets;
    for (const auto& test : m_testData)
        assets.push_back(test.first);

    kw::Fd1d<real> pricer;

    m_config.pdeCount = assets.size();
    ASSERT_EQ(pricer.allocate(m_config), "");

    std::vector<kw::Fd1dPde<real>> pdes;
    ASSERT_EQ(kw::Fd1dPdeFor(assets, m_config, pdes), "");

    ASSERT_EQ(pricer.solve(pdes), "");

    for (auto i = 0; i < m_testData.size(); ++i)
    {
        const auto& test = m_testData[i];
        for (const auto& [s, want] : test.second)
        {
            real got;
            ASSERT_EQ(pricer.value(i, s, got), "");
            EXPECT_NEAR(want, got, 1e-2);
        }
    }

    ASSERT_EQ(pricer.free(), "");
}


TEST_F(kwFd1dTest, Gpu)
{
    std::vector<kw::Option<real>> assets;
    for (const auto& test : m_testData)
        assets.push_back(test.first);


    kw::Fd1d_Gpu<real> pricer;

    m_config.pdeCount = assets.size();
    ASSERT_EQ(pricer.allocate(m_config), "");

    std::vector<kw::Fd1dPde<real>> pdes;
    ASSERT_EQ(kw::Fd1dPdeFor(assets, m_config, pdes), "");

    ASSERT_EQ(pricer.solve(pdes), "");

    for (auto i = 0; i < m_testData.size(); ++i)
    {
        const auto& test = m_testData[i];
        for (const auto& [s, want] : test.second)
        {
            real got;
            ASSERT_EQ(pricer.value(i, s, got), "");
            EXPECT_NEAR(want, got, 1e-2);
        }
    }

    ASSERT_EQ(pricer.free(), "");
}
