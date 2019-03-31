#include "stdafx.h"
#include "HyperworksElementCollectorFactory.hpp"

namespace aafc = axis::application::factories::collectors;
namespace aapc = axis::application::parsing::core;
namespace ada = axis::domain::analyses;
namespace asli = axis::services::language::iterators;
namespace aslp = axis::services::language::parsing;

aafc::HyperworksElementCollectorFactory::HyperworksElementCollectorFactory( void )
{
  factory_ = &aafc::GeneralElementCollectorFactory::Create();
}

aafc::HyperworksElementCollectorFactory::~HyperworksElementCollectorFactory( void )
{
  factory_->Destroy();
  factory_ = NULL;
}

void aafc::HyperworksElementCollectorFactory::Destroy( void ) const
{
  delete this;
}

aslp::ParseResult aafc::HyperworksElementCollectorFactory::TryParse( const axis::String& formatName, 
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

aafc::CollectorBuildResult aafc::HyperworksElementCollectorFactory::ParseAndBuild( 
                                                                     const axis::String& formatName, 
                                                                     const asli::InputIterator& begin, 
                                                                     const asli::InputIterator& end, 
                                                                     const ada::NumericalModel& model, 
                                                                     aapc::ParseContext& context )
{
  return factory_->ParseAndBuild(begin, end, model, context, builder_);
}

