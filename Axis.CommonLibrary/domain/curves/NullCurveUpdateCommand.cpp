#include "NullCurveUpdateCommand.hpp"

namespace adcu = axis::domain::curves;

adcu::NullCurveUpdateCommand::NullCurveUpdateCommand(void)
{
  // nothing to do here
}

adcu::NullCurveUpdateCommand::~NullCurveUpdateCommand(void)
{
  // nothing to do here
}

void axis::domain::curves::NullCurveUpdateCommand::Run( uint64, uint64, void *, 
  const axis::Dimension3D&, const axis::Dimension3D&, void * )
{
  // nothing to do  -- null implementation
}

adcu::CurveUpdateCommand& adcu::NullCurveUpdateCommand::GetInstance( void )
{
  static NullCurveUpdateCommand command;
  return command;
}