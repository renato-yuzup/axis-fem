#include "TextReportNodeCollectorFactory.hpp"
#include "TextReportNodeCollectorFactory_Pimpl.hpp"
#include "application/output/collectors/NodeDisplacementCollector.hpp"
#include "domain/analyses/NumericalModel.hpp"
#include "foundation/ArgumentException.hpp"
#include "application/output/collectors/Direction6DState.hpp"
#include "services/management/ServiceLocator.hpp"

namespace aafc = axis::application::factories::collectors;
namespace asli = axis::services::language::iterators;
namespace aaoc = axis::application::output::collectors;
namespace aapc = axis::application::parsing::core;
namespace ada = axis::domain::analyses;
namespace aslp = axis::services::language::parsing;
namespace asmg = axis::services::management;

aafc::TextReportNodeCollectorFactory::TextReportNodeCollectorFactory( void )
{
  pimpl_ = new Pimpl();
}

aafc::TextReportNodeCollectorFactory::~TextReportNodeCollectorFactory( void )
{
  delete pimpl_;
  pimpl_ = NULL;
}

void aafc::TextReportNodeCollectorFactory::Destroy( void ) const
{
  delete this;
}

aslp::ParseResult aafc::TextReportNodeCollectorFactory::TryParse( const axis::String& formatName, 
                                                                  const asli::InputIterator& begin, 
                                                                  const asli::InputIterator& end )
{
  if (formatName != _T("REPORT"))
  {
    aslp::ParseResult result;
    result.SetLastReadPosition(begin);
    result.SetResult(aslp::ParseResult::FailedMatch);
    return result;
  }
  return pimpl_->TryParseAny(begin, end);
}

aafc::CollectorBuildResult aafc::TextReportNodeCollectorFactory::ParseAndBuild( 
  const axis::String&, const asli::InputIterator& begin, const asli::InputIterator& end, 
  const ada::NumericalModel& model, aapc::ParseContext& context )
{
  return pimpl_->ParseAndBuildAny(begin, end, model, context);
}
