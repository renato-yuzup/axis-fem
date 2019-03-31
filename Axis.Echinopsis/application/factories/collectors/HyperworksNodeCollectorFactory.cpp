#include "stdafx.h"
#include "HyperworksNodeCollectorFactory.hpp"

namespace aafc = axis::application::factories::collectors;
namespace aapc = axis::application::parsing::core;
namespace ada = axis::domain::analyses;
namespace asli = axis::services::language::iterators;
namespace aslp = axis::services::language::parsing;

aafc::HyperworksNodeCollectorFactory::HyperworksNodeCollectorFactory( void )
{
  factory_ = &aafc::GeneralNodeCollectorFactory::Create();
}

aafc::HyperworksNodeCollectorFactory::~HyperworksNodeCollectorFactory( void )
{
  factory_->Destroy();
  factory_ = NULL;
}

void aafc::HyperworksNodeCollectorFactory::Destroy( void ) const
{
  delete this;
}

aslp::ParseResult aafc::HyperworksNodeCollectorFactory::TryParse( const axis::String& formatName, 
                                                                  const asli::InputIterator& begin, 
                                                                  const asli::InputIterator& end )
{
  if (formatName != _T("HYPERWORKS"))
  {
    aslp::ParseResult result;
    result.SetLastReadPosition(begin);
    result.SetResult(aslp::ParseResult::FailedMatch);
    return result;
  }
  return factory_->TryParse(begin, end);
}

aafc::CollectorBuildResult aafc::HyperworksNodeCollectorFactory::ParseAndBuild( 
                                                                  const axis::String& formatName, 
                                                                  const asli::InputIterator& begin, 
                                                                  const asli::InputIterator& end, 
                                                                  const ada::NumericalModel& model, 
                                                                  aapc::ParseContext& context )
{
  return factory_->ParseAndBuild(begin, end, model, context, builder_);
}
