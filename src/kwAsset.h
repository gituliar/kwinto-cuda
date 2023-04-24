#pragma once

#include <tuple>
#include <ostream>


namespace kw
{

enum class kParity { Call, Put };

template<typename Real>
struct Option
{
    Real
        t; // time to maturity (1.0 means 1 year)
    Real
        k; // strike (100 means $100)
    Real
        z; // volatility (0.2 means 20% per year)
    Real
        r; // interest rate (0.05 means 5% per year)
    Real
        q; // dividend rate (0.03 means 3% per year)
    bool
        e; // early exercise
    kParity
        w; // put or call

    friend bool operator<(const Option& l, const Option& r)
    {
        return std::tie(l.t, l.k, l.z, l.r, l.q, l.e, l.w)
            < std::tie(r.t, r.k, r.z, r.r, r.q, r.e, r.w);
    };
};

template<typename Real>
std::ostream& operator<<(std::ostream& os, const Option<Real>& o) {
    return os << "<Option t=" << o.t << ", k=" << o.k << ", z=" << o.z << ", r=" << o.r << ", q=" << o.q
        << ", " << (o.e ? "amer" : "euro") << ", " << (o.w == kParity::Call ? "call" : "put") << ">";
}

}