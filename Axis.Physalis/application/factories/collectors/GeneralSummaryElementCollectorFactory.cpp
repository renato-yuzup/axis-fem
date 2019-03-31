#include "GeneralSummaryElementCollectorFactory.hpp"
#include "GeneralSummaryElementCollectorFactory_Pimpl.hpp"
#include "GeneralSummaryElementCollectorBuilder.hpp"

namespace aafc = axis::application::factories::collectors;
namespace asli = axis::services::language::iterators;
namespace aapc = axis::application::parsing::core;
namespace ada = axis::domain::analyses;
namespace aslp = axis::services::language::parsing;

aafc::GeneralSummaryElementCollectorFactory::GeneralSummaryElementCollectorFactory( void )
{
  pimpl_ = new Pimpl();
}

aafc::GeneralSummaryElementCollectorFactory::~GeneralSummaryElementCollectorFactory( void )
{
  delete pimpl_;
  pimpl_ = NULL;
}

void aafc::GeneralSummaryElementCollectorFactory::Destroy( void ) const
{
  delete this;
}

aafc::GeneralSummaryElementCollectorFactory& aafc::GeneralSummaryElementCollectorFactory::Create( void )
{
  return *new GeneralSummaryElementCollectorFactory();
}

aslp::ParseResult aafc::GeneralSummaryElementCollectorFactory::TryParse(
                                                                      const asli::InputIterator& begin, 
                                                                      const asli::InputIterator& end )
{
  return pimpl_->TryParseAny(begin, end);
}

aafc::CollectorBuildResult aafc::GeneralSummaryElementCollectorFactory::ParseAndBuild( 
  const asli::InputIterator& begin, 
  const asli::InputIterator& end, 
  const ada::NumericalModel& model, 
  aapc::ParseContext& context )
{
  GeneralSummaryElementCollectorBuilder builder;
  return ParseAndBuild(begin, end, model, context, builder);
}

aafc::CollectorBuildResult aafc::GeneralSummaryElementCollectorFactory::ParseAndBuild( 
  const asli::InputIterator& begin, 
  const asli::InputIterator& end, 
  const ada::NumericalModel& model, 
  aapc::ParseContext& context,
  SummaryElementCollectorBuilder& builder)
{
  return pimpl_->ParseAndBuildAny(begin, end, model, context, builder);
}
