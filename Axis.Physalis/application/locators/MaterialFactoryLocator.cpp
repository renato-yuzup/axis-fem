#include "MaterialFactoryLocator.hpp"
#include "MaterialFactoryLocator_Pimpl.hpp"
#include "services/management/ServiceLocator.hpp"
#include "foundation/ArgumentException.hpp"
#include "foundation/InvalidOperationException.hpp"
#include "foundation/NotSupportedException.hpp"

namespace aafm = axis::application::factories::materials;
namespace aal = axis::application::locators;
namespace adm = axis::domain::materials;
namespace aslse = axis::services::language::syntax::evaluation;
namespace asmg = axis::services::management;

aal::MaterialFactoryLocator::MaterialFactoryLocator( void )
{
  pimpl_ = new Pimpl();
}

aal::MaterialFactoryLocator::~MaterialFactoryLocator(void)
{
  Pimpl::factory_set::iterator end = pimpl_->factories.end();
  for (Pimpl::factory_set::iterator it = pimpl_->factories.begin(); it != end; ++it)
  {
    (*it)->Destroy();
  }
  delete pimpl_;
}

void aal::MaterialFactoryLocator::RegisterFactory( aafm::MaterialFactory& factory )
{
  if (pimpl_->factories.find(&factory) != pimpl_->factories.end())
  {
    throw axis::foundation::ArgumentException();
  }
  pimpl_->factories.insert(&factory);
}

void aal::MaterialFactoryLocator::UnregisterFactory( aafm::MaterialFactory& factory )
{
  if (pimpl_->factories.find(&factory) == pimpl_->factories.end())
  {
    throw axis::foundation::ArgumentException();
  }
  pimpl_->factories.erase(&factory);
}

bool aal::MaterialFactoryLocator::CanBuild( const axis::String& materialName, 
                                            const aslse::ParameterList& params ) const
{
  // check if any factory is able to build this material
  Pimpl::factory_set::iterator end = pimpl_->factories.end();
  for (Pimpl::factory_set::iterator it = pimpl_->factories.begin(); it != end; ++it)
  {
    aafm::MaterialFactory& factory = **it;
    if (factory.CanBuild(materialName, params))
    {
      return true;
    }
  }

  // no one is able to build
  return false;
}

adm::MaterialModel& aal::MaterialFactoryLocator::BuildMaterial( const axis::String& materialName, 
                                                                const aslse::ParameterList& params )
{
  // check if any factory is able to build this material
  Pimpl::factory_set::iterator end = pimpl_->factories.end();
  for (Pimpl::factory_set::iterator it = pimpl_->factories.begin(); it != end; ++it)
  {
    aafm::MaterialFactory& factory = **it;
    if (factory.CanBuild(materialName, params))
    {
      return factory.Build(materialName, params);
    }
  }
  // no one could build this material type
  throw axis::foundation::InvalidOperationException();
}

const char * aal::MaterialFactoryLocator::GetFeaturePath( void ) const
{
  return asmg::ServiceLocator::GetMaterialFactoryLocatorPath();
}

const char * aal::MaterialFactoryLocator::GetFeatureName( void ) const
{
  return "MaterialFactoryLocator";
}
