#include "GeneralElementCollectorFactory.hpp"
#include "GeneralElementCollectorFactory_Pimpl.hpp"
#include "GeneralElementCollectorBuilder.hpp"

namespace aafc = axis::application::factories::collectors;
namespace asli = axis::services::language::iterators;
namespace aapc = axis::application::parsing::core;
namespace ada = axis::domain::analyses;
namespace aslp = axis::services::language::parsing;

aafc::GeneralElementCollectorFactory::GeneralElementCollectorFactory( void )
{
  pimpl_ = new Pimpl();
}

aafc::GeneralElementCollectorFactory::~GeneralElementCollectorFactory( void )
{
  delete pimpl_;
  pimpl_ = NULL;
}

void aafc::GeneralElementCollectorFactory::Destroy( void ) const
{
  delete this;
}

aafc::GeneralElementCollectorFactory& aafc::GeneralElementCollectorFactory::Create( void )
{
  return *new GeneralElementCollectorFactory();
}

aslp::ParseResult aafc::GeneralElementCollectorFactory::TryParse( const asli::InputIterator& begin, 
                                                                  const asli::InputIterator& end )
{
  return pimpl_->TryParseAny(begin, end);
}

aafc::CollectorBuildResult aafc::GeneralElementCollectorFactory::ParseAndBuild( 
                                                                    const asli::InputIterator& begin, 
                                                                    const asli::InputIterator& end, 
                                                                    const ada::NumericalModel& model, 
                                                                    aapc::ParseContext& context )
{
  GeneralElementCollectorBuilder builder;
  return ParseAndBuild(begin, end, model, context, builder);
}

aafc::CollectorBuildResult aafc::GeneralElementCollectorFactory::ParseAndBuild( 
                                                                    const asli::InputIterator& begin, 
                                                                    const asli::InputIterator& end, 
                                                                    const ada::NumericalModel& model, 
                                                                    aapc::ParseContext& context, 
                                                                    ElementCollectorBuilder& builder )
{
  return pimpl_->ParseAndBuildAny(begin, end, model, context, builder);
}
