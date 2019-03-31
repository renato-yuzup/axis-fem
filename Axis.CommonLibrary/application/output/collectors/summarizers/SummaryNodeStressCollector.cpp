#include "SummaryNodeStressCollector.hpp"
#include "domain/elements/Node.hpp"

namespace aaoc = axis::application::output::collectors;
namespace aaocs = axis::application::output::collectors::summarizers;
namespace ada = axis::domain::analyses;
namespace ade = axis::domain::elements;
namespace asmm = axis::services::messaging;

aaocs::SummaryNodeStressCollector::SummaryNodeStressCollector( const axis::String& targetSetName, 
                                                               SummaryType summaryType ) :
SummaryNode6DCollector(targetSetName, summaryType)
{
  // nothing to do here
}

aaocs::SummaryNodeStressCollector::SummaryNodeStressCollector( const axis::String& targetSetName, 
                                                               SummaryType summaryType, 
                                                               aaoc::XXDirectionState xxState, 
                                                               aaoc::YYDirectionState yyState, 
                                                               aaoc::ZZDirectionState zzState, 
                                                               aaoc::YZDirectionState yzState, 
                                                               aaoc::XZDirectionState xzState, 
                                                               aaoc::XYDirectionState xyState ) :
SummaryNode6DCollector(targetSetName, summaryType, xxState, yyState, zzState, 
  yzState, xzState, xyState)
{
  // nothing to do here
}

aaocs::SummaryNodeStressCollector::~SummaryNodeStressCollector( void )
{
  // nothing to do here
}

aaocs::SummaryNodeStressCollector& aaocs::SummaryNodeStressCollector::Create( 
  const axis::String& targetSetName, SummaryType summaryType )
{
  return *new aaocs::SummaryNodeStressCollector(targetSetName, summaryType);
}

aaocs::SummaryNodeStressCollector& aaocs::SummaryNodeStressCollector::Create( 
  const axis::String& targetSetName, SummaryType summaryType, aaoc::XXDirectionState xxState, 
  aaoc::YYDirectionState yyState, aaoc::ZZDirectionState zzState, 
  aaoc::YZDirectionState yzState, aaoc::XZDirectionState xzState, aaoc::XYDirectionState xyState )
{
  return *new aaocs::SummaryNodeStressCollector(targetSetName, summaryType, 
    xxState, yyState, zzState, yzState, xzState, xyState);
}

void aaocs::SummaryNodeStressCollector::Destroy( void ) const
{
  delete this;
}

real aaocs::SummaryNodeStressCollector::CollectValue( const asmm::ResultMessage&, 
                                                      const ade::Node& node, int directionIndex, 
                                                      const ada::NumericalModel& )
{
  return node.Stress()(directionIndex);
}

axis::String aaocs::SummaryNodeStressCollector::GetVariableName( bool plural ) const
{
  if (plural) return _T("stresses");
  return _T("stress");
}
