#include "GeneralNodeCollectorFactory.hpp"
#include "GeneralNodeCollectorFactory_Pimpl.hpp"
#include "GeneralNodeCollectorBuilder.hpp"

namespace aafc = axis::application::factories::collectors;
namespace asli = axis::services::language::iterators;
namespace aapc = axis::application::parsing::core;
namespace ada = axis::domain::analyses;
namespace aslp = axis::services::language::parsing;

aafc::GeneralNodeCollectorFactory::GeneralNodeCollectorFactory( void )
{
  pimpl_ = new Pimpl();
}

aafc::GeneralNodeCollectorFactory::~GeneralNodeCollectorFactory( void )
{
  delete pimpl_;
  pimpl_ = NULL;
}

void aafc::GeneralNodeCollectorFactory::Destroy( void ) const
{
  delete this;
}

aafc::GeneralNodeCollectorFactory& aafc::GeneralNodeCollectorFactory::Create( void )
{
  return *new GeneralNodeCollectorFactory();
}

aslp::ParseResult aafc::GeneralNodeCollectorFactory::TryParse( const asli::InputIterator& begin, 
                                                               const asli::InputIterator& end )
{
  return pimpl_->TryParseAny(begin, end);
}

aafc::CollectorBuildResult aafc::GeneralNodeCollectorFactory::ParseAndBuild( 
                                                                const asli::InputIterator& begin, 
                                                                const asli::InputIterator& end, 
                                                                const ada::NumericalModel& model, 
                                                                aapc::ParseContext& context )
{
  GeneralNodeCollectorBuilder builder;
  return ParseAndBuild(begin, end, model, context, builder);
}

aafc::CollectorBuildResult aafc::GeneralNodeCollectorFactory::ParseAndBuild( 
                                                                const asli::InputIterator& begin, 
                                                                const asli::InputIterator& end, 
                                                                const ada::NumericalModel& model, 
                                                                aapc::ParseContext& context,
                                                                NodeCollectorBuilder& builder)
{
  return pimpl_->ParseAndBuildAny(begin, end, model, context, builder);
}
