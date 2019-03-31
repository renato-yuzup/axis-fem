#pragma once
#include "foundation/Axis.Physalis.hpp"
#include "foundation/Axis.SystemBase.hpp"

namespace axis { namespace application { namespace parsing { namespace core {

class AXISPHYSALIS_API EntityLabeler
{
public:
  EntityLabeler(void);
  ~EntityLabeler(void);
  void Destroy(void) const;
  size_type PickNodeLabel(void);
  size_type PickElementLabel(void);
  size_type PickDofLabel(void);
  size_type GetGivenNodeLabelCount(void) const;
  size_type GetGivenElementLabelCount(void) const;
  size_type GetGivenDofLabelCount(void) const;
private:
  class Pimpl;
  Pimpl *pimpl_;
};

} } } } // namespace axis::application::parsing::core
