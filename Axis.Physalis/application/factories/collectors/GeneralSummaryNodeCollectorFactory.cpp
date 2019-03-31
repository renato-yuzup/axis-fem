#include "GeneralSummaryNodeCollectorFactory.hpp"
#include "GeneralSummaryNodeCollectorFactory_Pimpl.hpp"
#include "GeneralSummaryNodeCollectorBuilder.hpp"

namespace aafc = axis::application::factories::collectors;
namespace asli = axis::services::language::iterators;
namespace aaoc = axis::application::output::collectors;
namespace aaocs = axis::application::output::collectors::summarizers;
namespace aapc = axis::application::parsing::core;
namespace ada = axis::domain::analyses;
namespace aslp = axis::services::language::parsing;

aafc::GeneralSummaryNodeCollectorFactory::GeneralSummaryNodeCollectorFactory( void )
{
  pimpl_ = new Pimpl();
}

aafc::GeneralSummaryNodeCollectorFactory::~GeneralSummaryNodeCollectorFactory( void )
{
  delete pimpl_;
  pimpl_ = NULL;
}

void aafc::GeneralSummaryNodeCollectorFactory::Destroy( void ) const
{
  delete this;
}

aafc::GeneralSummaryNodeCollectorFactory& aafc::GeneralSummaryNodeCollectorFactory::Create( void )
{
  return *new GeneralSummaryNodeCollectorFactory();
}

aslp::ParseResult aafc::GeneralSummaryNodeCollectorFactory::TryParse(const asli::InputIterator& begin, 
                                                                     const asli::InputIterator& end )
{
  return pimpl_->TryParseAny(begin, end);
}

aafc::CollectorBuildResult aafc::GeneralSummaryNodeCollectorFactory::ParseAndBuild( 
                                                                const asli::InputIterator& begin, 
                                                                const asli::InputIterator& end, 
                                                                const ada::NumericalModel& model, 
                                                                aapc::ParseContext& context )
{
  GeneralSummaryNodeCollectorBuilder builder;
  return ParseAndBuild(begin, end, model, context, builder);
}

aafc::CollectorBuildResult aafc::GeneralSummaryNodeCollectorFactory::ParseAndBuild( 
                                                                const asli::InputIterator& begin, 
                                                                const asli::InputIterator& end, 
                                                                const ada::NumericalModel& model, 
                                                                aapc::ParseContext& context,
                                                                SummaryNodeCollectorBuilder& builder)
{
  return pimpl_->ParseAndBuildAny(begin, end, model, context, builder);
}
