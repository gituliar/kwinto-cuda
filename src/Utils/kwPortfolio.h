#pragma once

#include <vector>

#include "Core/kwAsset.h"
#include "Core/kwConfig.h"


namespace kw
{

class Portfolio
{
private:
    std::vector<Option>
        m_assets;
    std::vector<f64>
        m_prices;

public:
    /// Getters
    ///
    const auto&
        assets() const { return m_assets; }
    const auto&
        prices() const { return m_prices; }

    Error
        load(const std::string& srcPath);
    Error
        price(const Config& config, std::vector<f64>& prices);
    Error
        printPricesStats(const std::vector<f64>& prices);
};

}