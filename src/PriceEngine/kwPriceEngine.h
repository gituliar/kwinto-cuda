#pragma once

#include <vector>

#include "kwAsset.h"
#include "Utils/kwTypes.h"

namespace kw
{

class PriceEngine
{
public:
    virtual
    Error
        price(size_t i, double spot, double& price) const = 0;

    virtual
    Error
        run(const std::vector<Option>& assets) = 0;
};

}