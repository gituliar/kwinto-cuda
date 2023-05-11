#pragma once

#include <ostream>
#include <tuple>


namespace kw
{

enum class kParity { Call, Put };

struct Option
{
    double  t; // time to maturity (1.0 means 1 year)
    double  k; // strike (100 means $100)
    double  z; // volatility (0.2 means 20% per year)
    double  r; // interest rate (0.05 means 5% per year)
    double  q; // dividend rate (0.03 means 3% per year)
    double  s; // spot price (10 means $10)
    bool    e; // early exercise
    kParity w; // put or call

    friend bool operator<(const Option& l, const Option& r)
    {
        return std::tie(l.t, l.k, l.z, l.q, l.r, l.s, l.e, l.w)
            < std::tie(r.t, r.k, r.z, r.q, r.r, l.s, r.e, r.w);
    };
};

std::ostream&
    operator<<(std::ostream& os, const Option& o);

}