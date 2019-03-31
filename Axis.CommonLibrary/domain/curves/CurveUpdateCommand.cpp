#include "CurveUpdateCommand.hpp"

namespace adcu = axis::domain::curves;

adcu::CurveUpdateCommand::CurveUpdateCommand( void )
{
  time_ = 0;
}

adcu::CurveUpdateCommand::~CurveUpdateCommand( void )
{
  // nothing to do here
}

void adcu::CurveUpdateCommand::SetTime( real time )
{
  time_ = time;
}

real adcu::CurveUpdateCommand::GetTime( void ) const
{
  return time_;
}
