#pragma once

#include "kwAsset.h"
#include "Math/kwMath.h"
#include "Pricer/kwPricer.h"

namespace kw
{

class BlackScholes_Pricer : public Pricer
{
public:
    Error
        init(const Config& config) final;
    Error
        price(const std::vector<Option>& assets, std::vector<f64>& prices) final;

private:
    Error
        priceOne(const Option& asset, f64& price) const;
};

}