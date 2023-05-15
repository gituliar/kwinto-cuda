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
        price(size_t i, double spot, double& price) const = 0;

    virtual
    Error
        run(const std::vector<Option>& assets) = 0;
};

}