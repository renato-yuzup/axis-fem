#include "stdafx.h"
#include "NeoHookeanStrategy.hpp"

namespace adm = axis::domain::materials;

adm::NeoHookeanStrategy * adm::NeoHookeanStrategy::
  instance_ = new NeoHookeanStrategy();

adm::NeoHookeanStrategy::NeoHookeanStrategy(void)
{
  // nothing to do here
}

adm::NeoHookeanStrategy::~NeoHookeanStrategy(void)
{
  // nothing to do here
}

adm::UpdateStressCommand& adm::NeoHookeanStrategy::GetUpdateStressCommand( void )
{
  return stressCmd_;
}

adm::NeoHookeanStrategy& adm::NeoHookeanStrategy::GetInstance( void )
{
  return *instance_;
}
