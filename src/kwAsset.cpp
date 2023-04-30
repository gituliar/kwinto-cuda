#include "kwAsset.h"

std::ostream&
kw::operator<<(std::ostream& os, const kw::Option& o)
{
    return os << "<Option s=" << o.s << " t=" << o.t << ", k=" << o.k << ", z=" << o.z << ", r=" << o.r << ", q=" << o.q
        << ", " << (o.e ? "amer" : "euro") << ", " << (o.w == kw::kParity::Call ? "call" : "put") << ">";
}
