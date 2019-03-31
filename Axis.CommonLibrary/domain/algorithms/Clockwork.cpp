#include "Clockwork.hpp"

namespace adal = axis::domain::algorithms;
namespace ada = axis::domain::analyses;
namespace afm = axis::foundation::memory;

adal::Clockwork::Clockwork( void )
{
	// nothing to do here
}

adal::Clockwork::~Clockwork( void )
{
	// nothing to do here
}

bool adal::Clockwork::IsGPUCapable( void ) const
{
  return false;
}

void adal::Clockwork::CalculateNextTickOnGPU( ada::AnalysisTimeline&, const afm::RelativePointer&,
                                              adal::ExternalSolverFacade& )
{
  // nothing to do here in base implementation
}

void adal::Clockwork::CalculateNextTickOnGPU( ada::AnalysisTimeline&, const afm::RelativePointer&, 
                                              adal::ExternalSolverFacade&, real )
{
  // nothing to do here in base implementation
}
