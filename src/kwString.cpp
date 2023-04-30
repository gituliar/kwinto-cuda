#include "kwString.h"

#include <sstream>


std::vector<std::string>
kw::split(const std::string& src, char delim)
{
    std::vector<std::string> result;
    std::stringstream buf(src);

    for (std::string item; getline(buf, item, delim);) {
        result.push_back(item);
    }

    return result;
}
