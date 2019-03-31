#include "LinearElasticIsotropicStrategy.hpp"
#include "linear_iso_commands/LinearIsoStressCommand.hpp"

namespace adm = axis::domain::materials;

adm::LinearElasticIsotropicStrategy * 
  adm::LinearElasticIsotropicStrategy::instance_ = 
  new LinearElasticIsotropicStrategy();

adm::LinearElasticIsotropicStrategy::LinearElasticIsotropicStrategy(void)
{
  // nothing to do here
}

adm::LinearElasticIsotropicStrategy::~LinearElasticIsotropicStrategy(void)
{
  // nothing to do here
}

adm::UpdateStressCommand&
  adm::LinearElasticIsotropicStrategy::GetUpdateStressCommand( void )
{
  return stressCmd_;
}

adm::LinearElasticIsotropicStrategy& 
  adm::LinearElasticIsotropicStrategy::GetInstance( void )
{
  return *instance_;
}
