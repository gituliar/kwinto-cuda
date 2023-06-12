#pragma once

#include <ql/instruments/vanillaoption.hpp>

#include "kwPriceEngine.h"


namespace kw
{

constexpr auto DaysInYear = 365;


class Fd1dQl_PriceEngine : public PriceEngine
{
private:
    size_t
        m_tDim;
    size_t
        m_xDim;

public:
    virtual
    Error
        init(const Config& config) override;

    virtual
    Error
        price(const std::vector<Option>& assets, std::vector<double>& prices) override;

private:
    Error
        priceOne(const Option& asset, double& price) const;
};

}
