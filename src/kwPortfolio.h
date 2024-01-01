#pragma once

#include "Core/kwAsset.h"
#include "Core/kwString.h"

#include <map>


namespace kw
{

using Portfolio = std::map<kw::Option, double>; //  Option -> Price

kw::Error
    loadPortfolio(const std::string& srcPath_, Portfolio& portfolio);

}