#pragma once

#include <ostream>
#include <string>
#include <tuple>


namespace kw
{

enum class Parity { Call, Put };

struct Option
{
    double  t; //  time to maturity     1.0 = 365 days
    double  k; //  strike               100 = $100
    double  z; //  volatility           0.2 = 20% per year
    double  r; //  interest rate        0.05 = 5% per year
    double  q; //  dividend rate        0.03 = 3% per year
    double  s; //  spot price           100 = $100
    bool    e; //  early exercise       true = American, false = European
    Parity  w; //  put or call

    friend bool operator<(const Option& l, const Option& r)
    {
        return std::tie(l.t, l.k, l.z, l.q, l.r, l.s, l.e, l.w)
            < std::tie(r.t, r.k, r.z, r.q, r.r, r.s, r.e, r.w);
    };

    std::string asString() const
    {
        std::string res = "<Option";

        res += e ? " e=true" : " e=false";
        res += " k=" + std::to_string(k);
        res += " q=" + std::to_string(q);
        res += " r=" + std::to_string(r);
        res += " s=" + std::to_string(s);
        res += " t=" + std::to_string(t);
        res += w == Parity::Call ? " w=Call" : " w=Put";
        res += " z=" + std::to_string(z);
        res += ">";

        return res;
    }
};

std::ostream&
    operator<<(std::ostream& os, const Option& o);

}