#pragma once
#include "foundation/Axis.Physalis.hpp"
#include "services/management/Provider.hpp"
#include "application/factories/materials/MaterialFactory.hpp"
#include "domain/materials/MaterialModel.hpp"

namespace axis { namespace application { namespace locators {

class AXISPHYSALIS_API MaterialFactoryLocator: public axis::services::management::Provider
{
public:
  MaterialFactoryLocator(void);
	~MaterialFactoryLocator(void);
				
	void RegisterFactory(axis::application::factories::materials::MaterialFactory& factory);
	void UnregisterFactory(axis::application::factories::materials::MaterialFactory& factory);

	bool CanBuild(const axis::String& materialName, 
      const axis::services::language::syntax::evaluation::ParameterList& params) const;
	axis::domain::materials::MaterialModel& BuildMaterial(
      const axis::String& materialName, 
      const axis::services::language::syntax::evaluation::ParameterList& params);

  virtual const char * GetFeaturePath( void ) const;
  virtual const char * GetFeatureName( void ) const;
private:
  class Pimpl;
  Pimpl *pimpl_;
};			

} } } // namespace axis::application::locators
