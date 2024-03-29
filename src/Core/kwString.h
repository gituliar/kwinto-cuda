#pragma once

#include "Core/kwTypes.h"

#include <charconv>
#include <sstream>
#include <string>
#include <vector>


namespace kw
{

template<typename T>
Error
    fromString(const std::string_view& src, T& value)
{
    //if constexpr (std::is_same_v<T, double> || std::is_same_v<T, float>)
    //{
    //    if (fast_float::from_chars(src.data(), src.data() + src.size(), value).ec != std::errc{})
    //        return "kw::fromString : Fail to parse '" + std::string(src) + "'";
    //}
    //else
    //{
    //    if (std::from_chars(src.data(), src.data() + src.size(), value).ec != std::errc{})
    //        return "kw::fromString : Fail to parse '" + std::string(src) + "'";
    //}

    if (std::from_chars(src.data(), src.data() + src.size(), value).ec != std::errc{})
        return "kw::fromString : Fail to parse '" + std::string(src) + "'";

    return "";
};


static
std::vector<std::string>
    split(const std::string& src, char delim)
{
    std::vector<std::string> result;
    std::stringstream buf(src);

    for (std::string item; getline(buf, item, delim);) {
        result.push_back(item);
    }

    return result;
}

}
