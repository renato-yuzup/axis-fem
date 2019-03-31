#include "ElementSetCollection.hpp"

// Explicit instantiate class template
#include "foundation/collections/MappedSet.cpp"
template class axis::foundation::collections::MappedSet<axis::String, 
    axis::domain::collections::ElementSet>;
