#include "brolm/version.h"

#include <string>

namespace brolm {

const char* version_string() {
    // Built once from the version constants in version.h — single source of
    // truth, so the string can't drift from version_major/minor/patch.
    static const std::string s = std::to_string(version_major) + "." +
                                 std::to_string(version_minor) + "." +
                                 std::to_string(version_patch);
    return s.c_str();
}

}
