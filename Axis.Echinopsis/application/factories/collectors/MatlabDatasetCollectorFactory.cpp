#include "stdafx.h"
#include "MatlabDatasetCollectorFactory.hpp"

namespace aafc = axis::application::factories::collectors;
namespace aapc = axis::application::parsing::core;
namespace ada = axis::domain::analyses;
namespace aslp = axis::services::language::parsing;
namespace asli = axis::services::language::iterators;

aafc::MatlabDatasetCollectorFactory::MatlabDatasetCollectorFactory( void )
: sscnf_(aafc::GeneralSummaryNodeCollectorFactory::Create()), 
ssenf_(aafc::GeneralSummaryElementCollectorFactory::Create())
{
  // nothing to do here
}

aafc::MatlabDatasetCollectorFactory::~MatlabDatasetCollectorFactory( void )
{
  sscnf_.Destroy();
  ssenf_.Destroy();
}

void aafc::MatlabDatasetCollectorFactory::Destroy( void ) const
{
  delete this;
}

aslp::ParseResult aafc::MatlabDatasetCollectorFactory::TryParse( const axis::String& formatName, 
                                                                 const asli::InputIterator& begin, 
                                                                 const asli::InputIterator& end )
{
  if (formatName != _T("MATLAB_DATASET"))
  {
    aslp::ParseResult result;
    result.SetLastReadPosition(begin);
    result.SetResult(aslp::ParseResult::FailedMatch);
    return result;
  }
  aslp::ParseResult parseResult = sscnf_.TryParse(begin, end);
  if (parseResult.GetResult() == aslp::ParseResult::FailedMatch)
  { // failed; so we try the element parser collector
    parseResult = ssenf_.TryParse(begin, end);
  }
  return parseResult;
}

aafc::CollectorBuildResult aafc::MatlabDatasetCollectorFactory::ParseAndBuild( 
                                                        const axis::String& formatName, 
                                                        const asli::InputIterator& begin, 
                                                        const asli::InputIterator& end, 
                                                        const ada::NumericalModel& model, 
                                                        aapc::ParseContext& context )
{
  aslp::ParseResult parseResult = sscnf_.TryParse(begin, end);
  if (parseResult.IsMatch())
  { 
    return sscnf_.ParseAndBuild(begin, end, model, context);
  }

  // failed, try with element collector
  return ssenf_.ParseAndBuild(begin, end, model, context);
}

