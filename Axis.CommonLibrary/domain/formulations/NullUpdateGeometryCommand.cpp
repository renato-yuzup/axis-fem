#include "NullUpdateGeometryCommand.hpp"

namespace adf = axis::domain::formulations;
namespace afm = axis::foundation::memory;

adf::NullUpdateGeometryCommand::NullUpdateGeometryCommand( void ) 
{
  // nothing to do here
}

adf::NullUpdateGeometryCommand::~NullUpdateGeometryCommand( void )
{
  // nothing to do here
}

void adf::NullUpdateGeometryCommand::DoRun( uint64, uint64, void *, 
  const axis::Dimension3D&, const axis::Dimension3D&, void *, uint64, 
  afm::RelativePointer&, real, real, real )
{
  // nothing to do in null implementation
}
