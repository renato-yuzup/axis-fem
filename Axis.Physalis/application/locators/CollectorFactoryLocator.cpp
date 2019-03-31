#include "CollectorFactoryLocator.hpp"
#include "CollectorFactoryLocator_Pimpl.hpp"
#include "foundation/ArgumentException.hpp"
#include "foundation/definitions/AxisInputLanguage.hpp"
#include "services/management/ServiceLocator.hpp"
#include "application/parsing/parsers/ResultCollectorParser.hpp"
#include "foundation/InvalidOperationException.hpp"

namespace aaj = axis::application::jobs;
namespace aafc = axis::application::factories::collectors;
namespace aal = axis::application::locators;
namespace aapc = axis::application::parsing::core;
namespace aapps = axis::application::parsing::parsers;
namespace ada = axis::domain::analyses;
namespace aslp = axis::services::language::parsing;
namespace aslse = axis::services::language::syntax::evaluation;
namespace asli = axis::services::language::iterators;
namespace asmg = axis::services::management;
namespace afdf = axis::foundation::definitions;

aal::CollectorFactoryLocator::CollectorFactoryLocator( void )
{
  pimpl_ = new Pimpl();
  pimpl_->formatLocator = NULL;
}

aal::CollectorFactoryLocator::~CollectorFactoryLocator( void )
{
  Pimpl::factory_set::iterator end = pimpl_->builders.end();
  for (Pimpl::factory_set::iterator it = pimpl_->builders.begin(); it != end; ++it)
  {
    (**it).Destroy();
  }
	delete pimpl_;
  pimpl_ = NULL;
}

void aal::CollectorFactoryLocator::DoOnPostProcessRegistration( asmg::GlobalProviderCatalog& rootManager )
{
  pimpl_->formatLocator = &static_cast<WorkbookFactoryLocator&>(
        rootManager.GetProvider(asmg::ServiceLocator::GetWorkbookFactoryLocatorPath()));
}

void aal::CollectorFactoryLocator::RegisterFactory( aafc::CollectorFactory& factory )
{
  // check if we are not re-adding it
  if (pimpl_->builders.find(&factory) != pimpl_->builders.end())
  {
    throw axis::foundation::ArgumentException();
  }
  pimpl_->builders.insert(&factory);
}

void aal::CollectorFactoryLocator::UnregisterFactory( aafc::CollectorFactory& factory )
{
  // check if element exists
  Pimpl::factory_set::iterator it = pimpl_->builders.find(&factory);
  if (it == pimpl_->builders.end())
  {
    throw axis::foundation::ArgumentException();
  }
  pimpl_->builders.erase(it);
}

const char * aal::CollectorFactoryLocator::GetFeaturePath( void ) const
{
  return asmg::ServiceLocator::GetCollectorFactoryLocatorPath();
}

const char * aal::CollectorFactoryLocator::GetFeatureName( void ) const
{
  return "ResultCollectorBuilderLocator";
}

aslp::ParseResult aal::CollectorFactoryLocator::TryParse( const axis::String& formatName, 
                                                          const asli::InputIterator& begin, 
                                                          const asli::InputIterator& end )
{
  aslp::ParseResult bestResult;
  bestResult.SetResult(aslp::ParseResult::FailedMatch);
  bestResult.SetLastReadPosition(end);

  // search for the most suitable builder
  Pimpl::factory_set::iterator end_set = pimpl_->builders.end();
  for (Pimpl::factory_set::iterator it = pimpl_->builders.begin(); it != end_set; ++it)
  {
    Pimpl::factory_set::value_type builder = *it;
    aslp::ParseResult currentResult = builder->TryParse(formatName, begin, end);
    if (currentResult.IsMatch())
    {	// we found a best fit; stop here
      return currentResult;
    }
    else if (bestResult.GetResult() == aslp::ParseResult::FailedMatch && 
             currentResult.GetResult() != aslp::ParseResult::FailedMatch)
    {	// maybe if we request more input this one can fit in -- so far 
      // this seems to be the best; let's see the others
      bestResult = currentResult;
    }
  }
  return bestResult;
}

aafc::CollectorBuildResult aal::CollectorFactoryLocator::ParseAndBuild(
                                                            const ada::NumericalModel& model, 
                                                            aapc::ParseContext& context, 
                                                            const axis::String& formatName, 
                                                            const asli::InputIterator& begin, 
                                                            const asli::InputIterator& end)
{
  // we will use the first factory which claims to be able to parse
  Pimpl::factory_set::iterator end_set = pimpl_->builders.end();
  for (Pimpl::factory_set::iterator it = pimpl_->builders.begin(); it != end_set; ++it)
  {
    Pimpl::factory_set::value_type factory = *it;
    aslp::ParseResult result = factory->TryParse(formatName, begin, end);
    if (result.IsMatch())
    {	// we found a best fit; use this one
      return factory->ParseAndBuild(formatName, begin, end, model, context);
    }
  }
  // no builders found -- was TryParse used?
  throw axis::foundation::InvalidOperationException();
}

