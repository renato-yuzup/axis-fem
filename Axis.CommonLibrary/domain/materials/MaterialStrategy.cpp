#include "MaterialStrategy.hpp"
#include "NullMaterialStrategy.hpp"

namespace adm = axis::domain::materials;

adm::MaterialStrategy& adm::MaterialStrategy::NullStrategy = 
  *new NullMaterialStrategy();

adm::MaterialStrategy::MaterialStrategy(void)
{
  // nothing to do here
}

adm::MaterialStrategy::~MaterialStrategy(void)
{
  // nothing to do here
}
