#pragma once

#include <cmath>
#include <memory>
#include <string>


namespace kw
{

using Error = std::string;

using c8  = char;
using f32 = float;
using f64 = double;
using i8  = int8_t;
using i16 = int16_t;
using i32 = int32_t;
using i64 = int64_t;
using u8  = uint8_t;
using u16 = uint16_t;
using u32 = uint32_t;
using u64 = uint64_t;


using std::isnan;

constexpr f64
    nan = std::numeric_limits<f64>::quiet_NaN();

template<class T>
using sPtr = std::shared_ptr<T>;

template<class T, typename... Args>
sPtr<T>
make_sPtr(Args &&...args)
{
    return std::make_shared<T>(args...);
};

}
