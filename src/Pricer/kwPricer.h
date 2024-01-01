#pragma once

#include "kwAsset.h"
#include "Core/kwTypes.h"
#include "Utils/kwConfig.h"

#include <vector>


namespace kw
{

class Pricer
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