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
        t; // time to maturity
    Real
        k; // strike
    Real
        z; // volatility
    Real
        r; // interest rate
    Real
        q; // dividend rate
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
        << ", " << (o.e ? "amer" : "euro") << ", " << (o.w == kParity::Call ? "call" : "p") << ">";
}

}