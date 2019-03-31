#include "WorkbookFactoryLocator.hpp"
#include "WorkbookFactoryLocator_Pimpl.hpp"
#include "application/factories/workbooks/WorkbookFactory.hpp"
#include "services/management/ServiceLocator.hpp"
#include "foundation/ArgumentException.hpp"
#include "foundation/InvalidOperationException.hpp"

namespace aal = axis::application::locators;
namespace aaff = axis::application::factories::workbooks;
namespace aaow = axis::application::output::workbooks;
namespace aslse = axis::services::language::syntax::evaluation;
namespace asmg = axis::services::management;

aal::WorkbookFactoryLocator::WorkbookFactoryLocator( void )
{
	pimpl_ = new Pimpl();
}

aal::WorkbookFactoryLocator::~WorkbookFactoryLocator( void )
{
  Pimpl::factory_set::iterator end = pimpl_->factories.end();
  for (Pimpl::factory_set::iterator it = pimpl_->factories.begin(); it != end; ++it)
  {
    (**it).Destroy();
  }
	delete pimpl_;
}

void aal::WorkbookFactoryLocator::RegisterFactory( aaff::WorkbookFactory& factory )
{
  // check for duplicates
  if (pimpl_->factories.find(&factory) != pimpl_->factories.end())
  {
    throw axis::foundation::ArgumentException();
  }
  pimpl_->factories.insert(&factory);
}

void aal::WorkbookFactoryLocator::UnregisterFactory( aaff::WorkbookFactory& factory )
{
  // check for existence
  Pimpl::factory_set::iterator where = pimpl_->factories.find(&factory);
  if (where == pimpl_->factories.end())
  {
    throw axis::foundation::ArgumentException();
  }
  pimpl_->factories.erase(where);
}

bool aal::WorkbookFactoryLocator::CanBuild( const axis::String& formatName, 
                                            const aslse::ParameterList& formatArguments ) const
{
  Pimpl::factory_set::iterator end = pimpl_->factories.end();
  for (Pimpl::factory_set::iterator it = pimpl_->factories.begin(); it != end; ++it)
  {
    aaff::WorkbookFactory& factory = **it;
    if (factory.CanBuild(formatName, formatArguments))
    {
      return true;
    }
  }
  return false;
}

aaow::ResultWorkbook& aal::WorkbookFactoryLocator::BuildWorkbook( const axis::String& formatName, 
    const aslse::ParameterList& formatArguments )
{
  Pimpl::factory_set::iterator end = pimpl_->factories.end();
  for (Pimpl::factory_set::iterator it = pimpl_->factories.begin(); it != end; ++it)
  {
    aaff::WorkbookFactory& factory = **it;
    if (factory.CanBuild(formatName, formatArguments))
    {
      return factory.BuildWorkbook(formatName, formatArguments);
    }
  }
  throw axis::foundation::InvalidOperationException(
      _T("No factories claimed sufficient knowledge to build the specified format."));
}

const char * aal::WorkbookFactoryLocator::GetFeaturePath( void ) const
{
  return asmg::ServiceLocator::GetWorkbookFactoryLocatorPath();
}

const char * aal::WorkbookFactoryLocator::GetFeatureName( void ) const
{
  return "StandardFormatFactoryLocator";
}
