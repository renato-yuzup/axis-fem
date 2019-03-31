#include "FormulationStrategy.hpp"
#include "NullFormulationStrategy.hpp"

namespace adf = axis::domain::formulations;

adf::FormulationStrategy& adf::FormulationStrategy::NullStrategy = 
  *new NullFormulationStrategy();

adf::FormulationStrategy::FormulationStrategy(void)
{
  // nothing to do here
}


adf::FormulationStrategy::~FormulationStrategy(void)
{
  // nothing to do here
}
