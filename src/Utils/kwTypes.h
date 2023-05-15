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


// Taken from https://gitlab.com/libeigen/eigen/-/blob/master/Eigen/src/Core/util/ForwardDeclarations.h
template<typename T> struct traits;
template<typename T> struct traits<const T> : traits<T> {};

}
