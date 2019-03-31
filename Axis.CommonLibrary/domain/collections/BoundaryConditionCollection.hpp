#pragma once
#include "foundation/Axis.CommonLibrary.hpp"
#include "foundation/collections/AssociativeCollection.hpp"
#include "domain/elements/DoF.hpp"
#include "domain/boundary_conditions/BoundaryCondition.hpp"

namespace axis { namespace domain	{ namespace collections {

typedef AXISCOMMONLIBRARY_API
    axis::foundation::collections::AssociativeCollection<axis::domain::elements::DoF, 
        axis::domain::boundary_conditions::BoundaryCondition>
    BoundaryConditionCollection;

} } } // namespace axis::domain::collections

