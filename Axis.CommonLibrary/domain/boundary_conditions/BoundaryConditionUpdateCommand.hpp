#pragma once
#include "foundation/computing/KernelCommand.hpp"
#include "foundation/Axis.CommonLibrary.hpp"
#include "foundation/memory/RelativePointer.hpp"

namespace axis { namespace domain { namespace boundary_conditions {

class AXISCOMMONLIBRARY_API BoundaryConditionUpdateCommand : public axis::foundation::computing::KernelCommand
{
public:
  BoundaryConditionUpdateCommand(void);
  virtual ~BoundaryConditionUpdateCommand(void);
  void Configure(real time, axis::foundation::memory::RelativePointer& vectorMaskPtr);
protected:
  real GetTime(void) const;
  axis::foundation::memory::RelativePointer GetVectorMask(void) const;
private:
  real time_;
  axis::foundation::memory::RelativePointer vectorMaskPtr_;
};

} } } // namespace axis::domain::boundary_conditions
