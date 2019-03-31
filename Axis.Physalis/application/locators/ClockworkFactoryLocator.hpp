#pragma once
#include "foundation/Axis.Physalis.hpp"
#include "services/management/Provider.hpp"
#include "application/factories/algorithms/ClockworkFactory.hpp"

namespace axis { namespace application { namespace locators {

class AXISPHYSALIS_API ClockworkFactoryLocator : public axis::services::management::Provider
{
public:
  ClockworkFactoryLocator(void);
	~ClockworkFactoryLocator(void);
	void Destroy(void) const;
	bool CanBuild(const axis::String& clockworkTypeName, 
                const axis::services::language::syntax::evaluation::ParameterList& paramList, 
                real stepStartTime, real stepEndTime) const;
	axis::domain::algorithms::Clockwork& BuildClockwork(const axis::String& clockworkTypeName, 
    const axis::services::language::syntax::evaluation::ParameterList& paramList, 
    real stepStartTime, real stepEndTime);
	void RegisterFactory(axis::application::factories::algorithms::ClockworkFactory& factory);
	void UnregisterFactory(axis::application::factories::algorithms::ClockworkFactory& factory);
  virtual const char * GetFeaturePath( void ) const;
  virtual const char * GetFeatureName( void ) const;
private:
  class Pimpl;
  Pimpl *pimpl_;
};

} } } // namespace axis::application::locators
