#pragma once

#include "kwAsset.h"
#include "Utils/kwConfig.h"
#include "Utils/kwTypes.h"

#include <vector>


namespace kw
{

class PriceEngine
{
public:
    virtual
    Error
        init(const Config& config) = 0;

    virtual
    Error
        price(const std::vector<Option>& assets, std::vector<double>& prices) = 0;
};

}