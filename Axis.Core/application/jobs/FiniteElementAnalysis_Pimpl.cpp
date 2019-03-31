#include "FiniteElementAnalysis_Pimpl.hpp"

axis::application::jobs::FiniteElementAnalysis::Pimpl::Pimpl( void )
{
  model = NULL;
  solver = NULL;
  timeline = NULL;
  resultArchives = NULL;
  monitor = NULL;
  postProcessor = NULL;
  workFolder = NULL;
  stepIndex = -1;
}
