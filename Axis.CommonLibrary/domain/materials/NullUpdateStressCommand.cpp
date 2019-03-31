#include "NullUpdateStressCommand.hpp"

namespace adm = axis::domain::materials;

adm::NullUpdateStressCommand::NullUpdateStressCommand(void)
{
  // nothing to do here
}

adm::NullUpdateStressCommand::~NullUpdateStressCommand(void)
{
  // nothing to do here
}

void adm::NullUpdateStressCommand::DoRun( uint64, uint64, void *, 
  const axis::Dimension3D&, const axis::Dimension3D&, void *, uint64, 
  axis::foundation::memory::RelativePointer&, real, real, real )
{
  // nothing to do here
}
