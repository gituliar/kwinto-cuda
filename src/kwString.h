#pragma once

#include <charconv>

#include "kwCommon.h"

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

}
