#include "NullMaterialStrategy.hpp"
#include "NullUpdateStressCommand.hpp"

namespace adm = axis::domain::materials;

adm::NullMaterialStrategy::NullMaterialStrategy(void)
{
  stressCommand_ = new NullUpdateStressCommand();
}


adm::NullMaterialStrategy::~NullMaterialStrategy(void)
{
  delete stressCommand_;
}

adm::UpdateStressCommand& 
  adm::NullMaterialStrategy::GetUpdateStressCommand( void )
{
  return *stressCommand_;
}
