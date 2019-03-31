#include "ConstraintParserLocator.hpp"
#include "ConstraintParserLocator_Pimpl.hpp"
#include "foundation/ArgumentException.hpp"
#include "foundation/definitions/AxisInputLanguage.hpp"
#include "services/management/ServiceLocator.hpp"
#include "application/parsing/parsers/ConstraintParser.hpp"
#include "foundation/InvalidOperationException.hpp"
#include "application/factories/boundary_conditions/ConstraintFactory.hpp"

namespace aapps = axis::application::parsing::parsers;
namespace aaj = axis::application::jobs;
namespace aapc = axis::application::parsing::core;
namespace afd = axis::foundation::definitions;
namespace asmg = axis::services::management;
namespace asli = axis::services::language::iterators;
namespace aslp = axis::services::language::parsing;
namespace aslse = axis::services::language::syntax::evaluation;
namespace aafbc = axis::application::factories::boundary_conditions;
namespace aal = axis::application::locators;

aal::ConstraintParserLocator::ConstraintParserLocator( void )
{
  pimpl_ = new Pimpl();
}

aal::ConstraintParserLocator::~ConstraintParserLocator( void )
{
  Pimpl::builder_iterator end = pimpl_->builders.end();
  for (Pimpl::builder_iterator it = pimpl_->builders.begin(); it != end; ++it)
  {
    aafbc::ConstraintFactory& factory = **it;
  }
	delete pimpl_;
}

void aal::ConstraintParserLocator::RegisterFactory( aafbc::ConstraintFactory& builder )
{
  // check if we are not re-adding it
  if (pimpl_->builders.find(&builder) != pimpl_->builders.end())
  {
    throw axis::foundation::ArgumentException();
  }
  pimpl_->builders.insert(&builder);
}

void aal::ConstraintParserLocator::UnregisterFactory( aafbc::ConstraintFactory& builder )
{
  // check if we are not re-adding it
  Pimpl::builder_iterator it = pimpl_->builders.find(&builder);
  if (it == pimpl_->builders.end())
  {
    throw axis::foundation::ArgumentException();
  }
  pimpl_->builders.erase(it);
}

bool aal::ConstraintParserLocator::CanParse( const axis::String& blockName, 
                                             const aslse::ParameterList& paramList )
{
  return (blockName == afd::AxisInputLanguage::ConstraintBlockName) && paramList.IsEmpty();
}

aapps::BlockParser& aal::ConstraintParserLocator::BuildParser( const axis::String& contextName, 
                                                               const aslse::ParameterList& paramList )
{
  if (!CanParse(contextName, paramList))
  {
    throw axis::foundation::InvalidOperationException();
  }
  return *new aapps::ConstraintParser(*this);
}

const char * aal::ConstraintParserLocator::GetFeaturePath( void ) const
{
  return asmg::ServiceLocator::GetNodalConstraintParserLocatorPath();
}

const char * aal::ConstraintParserLocator::GetFeatureName( void ) const
{
  return "ConstraintParserLocator";
}

aslp::ParseResult aal::ConstraintParserLocator::TryParse( const asli::InputIterator& begin, 
                                                          const asli::InputIterator& end )
{
  aslp::ParseResult bestResult;
  bestResult.SetResult(aslp::ParseResult::FailedMatch);
  bestResult.SetLastReadPosition(begin);

  // search for the most suitable builder
  Pimpl::builder_iterator end_set = pimpl_->builders.end();
  for (Pimpl::builder_iterator it = pimpl_->builders.begin(); it != end_set; ++it)
  {
    Pimpl::builder_set::value_type builder = *it;
    aslp::ParseResult currentResult = builder->TryParse(begin, end);
    if (bestResult.GetResult() == aslp::ParseResult::FailedMatch && 
        currentResult.GetResult() != aslp::ParseResult::FailedMatch)
    {	// maybe if we request more input this one can fit it -- so far this 
      // seems to be the best; let's see the others
      bestResult = currentResult;
    }
    else if (bestResult.GetResult() == aslp::ParseResult::FullReadPartialMatch 
             && currentResult.IsMatch())
    {	// we found a best fit; stop here
      return currentResult;
    }
  }

  return bestResult;
}

aslp::ParseResult aal::ConstraintParserLocator::ParseAndBuild( aaj::StructuralAnalysis& analysis, 
                                                               aapc::ParseContext& context, 
                                                               const asli::InputIterator& begin, 
                                                               const asli::InputIterator& end )
{
  //we will use the first builder which claims to be able to parse
  Pimpl::builder_iterator end_set = pimpl_->builders.end();
  for (Pimpl::builder_iterator it = pimpl_->builders.begin(); it != end_set; ++it)
  {
    Pimpl::builder_set::value_type builder = *it;
    aslp::ParseResult result = builder->TryParse(begin, end);
    if (result.IsMatch())
    {	// we found a best fit; use this one
      return builder->ParseAndBuild(analysis, context, begin, end);
    }
  }
  // no builders found -- was TryParse used?
  throw axis::foundation::InvalidOperationException();
}
