#pragma once
#include "foundation/collections/MappedSet.hpp"
#include "AxisString.hpp"
#include "NodeSet.hpp"

namespace axis { namespace domain { namespace collections {

  typedef AXISCOMMONLIBRARY_API axis::foundation::collections::MappedSet<axis::String,
                                                     axis::domain::collections::NodeSet>
      NodeSetCollection;

} } } // namespace axis::domain::collections
