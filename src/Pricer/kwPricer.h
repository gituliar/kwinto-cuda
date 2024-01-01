#pragma once

#include "Core/kwAsset.h"
#include "Core/kwConfig.h"

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