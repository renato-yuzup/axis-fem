#include "StructuralAnalysis_Pimpl.hpp"

axis::application::jobs::StructuralAnalysis::Pimpl::Pimpl( void )
{
  model = NULL;
  workFolder = NULL;
  currentStepIndex = -1;
}

axis::application::jobs::StructuralAnalysis::Pimpl::~Pimpl( void )
{
  // nothing to do here
}
