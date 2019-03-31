#pragma once
#include "foundation/Axis.CommonLibrary.hpp"
#include "domain/curves/Curve.hpp"
#include "foundation/collections/RelativeMap.hpp"
#include "AxisString.hpp"

namespace axis { namespace domain { namespace collections {

typedef AXISCOMMONLIBRARY_API axis::foundation::collections::RelativeMap<
    axis::String, axis::domain::curves::Curve> CurveSet;

} } } // namespace axis::domain::collections

