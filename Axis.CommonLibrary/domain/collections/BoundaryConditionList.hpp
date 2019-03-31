#pragma once
#include "foundation/Axis.CommonLibrary.hpp"
#include "domain/boundary_conditions/BoundaryCondition.hpp"
#include "foundation/collections/List.hpp"

namespace axis { namespace domain { namespace collections	{

typedef AXISCOMMONLIBRARY_API
    axis::foundation::collections::List<
        axis::domain::boundary_conditions::BoundaryCondition>
    BoundaryConditionList;

} } } // namespace axis::domain::collections
