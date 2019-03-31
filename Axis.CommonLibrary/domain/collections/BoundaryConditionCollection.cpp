#include "BoundaryConditionCollection.hpp"

// Explicit instantiate class template
#include "foundation/collections/AssociativeCollection.cpp"
template class axis::foundation::collections::AssociativeCollection<
    axis::domain::elements::DoF, 
    axis::domain::boundary_conditions::BoundaryCondition>;
