#include "NullInternalForceCommand.hpp"

namespace adf = axis::domain::formulations;

adf::NullInternalForceCommand::NullInternalForceCommand(void)
{
  // nothing to do here
}

adf::NullInternalForceCommand::~NullInternalForceCommand(void)
{
  // nothing to do here
}

void adf::NullInternalForceCommand::DoRun( uint64, uint64, void *, 
  const axis::Dimension3D&, const axis::Dimension3D&, void *, uint64, 
  axis::foundation::memory::RelativePointer&, real, real, real )
{
  // nothing to do here
}
