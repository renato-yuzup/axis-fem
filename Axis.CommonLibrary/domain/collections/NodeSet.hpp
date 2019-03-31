#pragma once
#include "foundation/Axis.CommonLibrary.hpp"
#include "domain/elements/Node.hpp"
#include "foundation/collections/BimapSet.hpp"

namespace axis{ namespace domain { namespace collections {

typedef AXISCOMMONLIBRARY_API axis::foundation::collections::BimapSet<
    axis::domain::elements::Node> NodeSet;

} } }
