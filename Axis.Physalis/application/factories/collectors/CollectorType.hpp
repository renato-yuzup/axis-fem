#pragma once
#include "foundation/Axis.Physalis.hpp"

namespace axis { namespace application { namespace factories { namespace collectors {

/**
 * Indicates the collector type by its target data.
 */
enum AXISPHYSALIS_API CollectorType
{
  kDisplacement,
  kVelocity,
  kAcceleration,
  kReactionForce,
  kExternalLoad,
  kStress,
  kStrain,
  kArtificialEnergy,
  kEffectivePlasticStrain,
  kPlasticStrain,
  kDeformationGradient,
  kUndefined
};

} } } } // namespace axis::application::factories::collectors
