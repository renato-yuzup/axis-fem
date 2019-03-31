#pragma once
#include "foundation/Axis.CommonLibrary.hpp"
#include "domain/elements/FiniteElement.hpp"
#include "foundation/collections/BimapSet.hpp"

namespace axis { namespace domain	{ namespace collections	{		

typedef AXISCOMMONLIBRARY_API axis::foundation::collections::BimapSet<
    axis::domain::elements::FiniteElement> ElementSet;

} } } // namespace axis::domain::collections
