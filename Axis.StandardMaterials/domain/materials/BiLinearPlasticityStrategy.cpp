#include "stdafx.h"
#include "BiLinearPlasticityStrategy.hpp"

namespace adm = axis::domain::materials;

adm::BiLinearPlasticityStrategy * adm::BiLinearPlasticityStrategy::
  instance_ = new adm::BiLinearPlasticityStrategy();

adm::BiLinearPlasticityStrategy::BiLinearPlasticityStrategy(void)
{
  // nothing to do here
}

adm::BiLinearPlasticityStrategy::~BiLinearPlasticityStrategy(void)
{
  // nothing to do here
}

adm::UpdateStressCommand& 
  adm::BiLinearPlasticityStrategy::GetUpdateStressCommand( void )
{
  return stressCmd_;
}

adm::BiLinearPlasticityStrategy& 
  adm::BiLinearPlasticityStrategy::GetInstance( void )
{
  return *instance_;
}
