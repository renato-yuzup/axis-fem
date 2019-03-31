#include "CurveSet.hpp"

// Explicit instantiate class template
#include "foundation/collections/RelativeMap.cpp"
template class axis::foundation::collections::RelativeMap<
    axis::String, axis::domain::curves::Curve>;
