#pragma once

#include "kwAsset.h"
#include "kwString.h"

#include <map>


namespace kw
{

using Portfolio = std::map<kw::Option, double>; //  Option -> Price

kw::Error
    loadPortfolio(const std::string& srcPath_, Portfolio& portfolio);

}