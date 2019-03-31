#include "AxisApplication.hpp"
#include "application/runnable/AxisApplicationFacade.hpp"

namespace aar = axis::application::runnable;

aar::AxisApplication::AxisApplication(void)
{
	// nothing to do here
}

aar::AxisApplication::~AxisApplication(void)
{
	// nothing to do here
}

aar::AxisApplication& aar::AxisApplication::CreateApplication( void )
{
	return *new aar::AxisApplicationFacade();
}
