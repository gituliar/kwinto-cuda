#pragma once

#include <memory>
#include <string>


namespace kw
{

using Error = std::string;


template<class T>
using sPtr = std::shared_ptr<T>;

template<class T, typename... Args>
sPtr<T>
make_sPtr(Args &&...args)
{
    return std::make_shared<T>(args...);
};

}
