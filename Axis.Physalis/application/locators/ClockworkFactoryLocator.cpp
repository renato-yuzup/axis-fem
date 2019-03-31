#include "ClockworkFactoryLocator.hpp"
#include "ClockworkFactoryLocator_Pimpl.hpp"
#include "foundation/InvalidOperationException.hpp"
#include "services/management/ServiceLocator.hpp"

namespace aafa = axis::application::factories::algorithms;
namespace aal = axis::application::locators;
namespace adal = axis::domain::algorithms;
namespace asmg = axis::services::management;
namespace aslse = axis::services::language::syntax::evaluation;
namespace afc = axis::foundation::collections;

aal::ClockworkFactoryLocator::ClockworkFactoryLocator( void )
{
	pimpl_ = new Pimpl();
}

aal::ClockworkFactoryLocator::~ClockworkFactoryLocator( void )
{
  // destroy all registered factories
  Pimpl::factory_iterator end = pimpl_->factories.end();
  for (Pimpl::factory_iterator it = pimpl_->factories.begin(); it != end; ++it)
  {
    (**it).Destroy();
  }
  delete pimpl_;
  pimpl_ = NULL;
}

void aal::ClockworkFactoryLocator::Destroy( void ) const
{
  delete this;
}

bool aal::ClockworkFactoryLocator::CanBuild( const axis::String& clockworkTypeName, 
                                             const aslse::ParameterList& paramList, 
                                             real stepStartTime, real stepEndTime ) const
{
  Pimpl::factory_iterator end = pimpl_->factories.end();
  for (Pimpl::factory_iterator it = pimpl_->factories.begin(); it != end; ++it)
  {
    aafa::ClockworkFactory& f = **it;
    if (f.CanBuild(clockworkTypeName, paramList, stepStartTime, stepEndTime))
    {
      return true;			
    }
  }
  return false;
}

adal::Clockwork& aal::ClockworkFactoryLocator::BuildClockwork( const axis::String& clockworkTypeName, 
                                                               const aslse::ParameterList& paramList, 
                                                               real stepStartTime, real stepEndTime )
{
  Pimpl::factory_iterator end = pimpl_->factories.end();
  for (Pimpl::factory_iterator it = pimpl_->factories.begin(); it != end; ++it)
  {
    aafa::ClockworkFactory& f = **it;
    if (f.CanBuild(clockworkTypeName, paramList, stepStartTime, stepEndTime))
    {
      return f.Build(clockworkTypeName, paramList, stepStartTime, stepEndTime);			
    }
  }
  // could not build clockwork
  throw axis::foundation::InvalidOperationException(_T("No factories available to build specified type."));
}

void aal::ClockworkFactoryLocator::RegisterFactory( aafa::ClockworkFactory& factory )
{
  pimpl_->factories.insert(&factory);
}

void aal::ClockworkFactoryLocator::UnregisterFactory( aafa::ClockworkFactory& factory )
{
  if (pimpl_->factories.find(&factory) == pimpl_->factories.end())
  {
    throw axis::foundation::ElementNotFoundException();
  }
  pimpl_->factories.erase(&factory);
}

const char * aal::ClockworkFactoryLocator::GetFeaturePath( void ) const
{
  return asmg::ServiceLocator::GetClockworkFactoryLocatorPath();
}

const char * aal::ClockworkFactoryLocator::GetFeatureName( void ) const
{
  return "AxisClockworkFactoryLocator";
}
