#pragma once
#include "foundation/Axis.Physalis.hpp"


namespace axis { namespace application { namespace factories { namespace collectors {

/**
 * Indicates on which type of entity a collector acts.
 */
enum AXISPHYSALIS_API EntityType
{
  kGenericCollectorType,
  kNodeCollectorType,
  kElementCollectorType
};

} } } } // namespace axis::application::factories::collectors
