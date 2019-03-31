#include "NonLinearHexaFBReducedStrategy.hpp"

namespace adf = axis::domain::formulations;

adf::NonLinearHexaFBReducedStrategy * adf::NonLinearHexaFBReducedStrategy
	::instance_ = new NonLinearHexaFBReducedStrategy();

adf::NonLinearHexaFBReducedStrategy::NonLinearHexaFBReducedStrategy(void)
{
	// nothing to do here
}

adf::NonLinearHexaFBReducedStrategy::~NonLinearHexaFBReducedStrategy(void)
{
	// nothing to do here
}

adf::NonLinearHexaFBReducedStrategy& 
	adf::NonLinearHexaFBReducedStrategy::GetInstance( void )
{
	return *instance_;
}

adf::UpdateStrainCommand& 
	adf::NonLinearHexaFBReducedStrategy::GetUpdateStrainCommand( void )
{
	return updateStrainCmd_;
}

adf::InternalForceCommand& 
	adf::NonLinearHexaFBReducedStrategy::GetUpdateInternalForceStrategy( void )
{
	return internalForceCmd_;
}

adf::UpdateGeometryCommand& 
	adf::NonLinearHexaFBReducedStrategy::GetUpdateGeometryCommand( void )
{
	return updateGeomCommand_;
}
