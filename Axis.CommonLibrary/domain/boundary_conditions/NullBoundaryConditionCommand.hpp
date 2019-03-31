#pragma once
#include "BoundaryConditionUpdateCommand.hpp"
#include "foundation/Axis.CommonLibrary.hpp"

namespace axis { namespace domain { namespace boundary_conditions {

class AXISCOMMONLIBRARY_API NullBoundaryConditionCommand : public BoundaryConditionUpdateCommand
{
public:
  NullBoundaryConditionCommand(void);
  ~NullBoundaryConditionCommand(void);
  virtual void Run( uint64, uint64, void *, const axis::Dimension3D&, 
    const axis::Dimension3D&, void * );

  static BoundaryConditionUpdateCommand& GetInstance(void);
};

} } } // namespace axis::domain::boundary_conditions
