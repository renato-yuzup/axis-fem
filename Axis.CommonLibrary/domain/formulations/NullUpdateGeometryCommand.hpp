#pragma once
#include "foundation/Axis.CommonLibrary.hpp"
#include "UpdateGeometryCommand.hpp"

namespace axis { namespace domain { namespace formulations {

class AXISCOMMONLIBRARY_API NullUpdateGeometryCommand : 
  public UpdateGeometryCommand
{
public:
  NullUpdateGeometryCommand(void);
  ~NullUpdateGeometryCommand(void);
private:
  virtual void DoRun( uint64, uint64, void *, const axis::Dimension3D&, 
    const axis::Dimension3D&, void *, uint64, 
    axis::foundation::memory::RelativePointer&, real, real, real );
};

} } } // namespace axis::domain::formulations
