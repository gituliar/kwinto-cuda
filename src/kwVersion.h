#include <string>

namespace kw
{

class Version
{
public:
    static const std::string BUILD_DATE;
    static const std::string GIT_BRANCH;
    static const std::string GIT_DATE;
    static const std::string GIT_DESCRIBE;
    static const std::string GIT_REV;
    static const std::string GIT_TAG;
};

}
