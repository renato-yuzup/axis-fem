#include "SummaryNodeStrainCollector.hpp"
#include "domain/elements/Node.hpp"

namespace aaoc = axis::application::output::collectors;
namespace aaocs = axis::application::output::collectors::summarizers;
namespace ada = axis::domain::analyses;
namespace ade = axis::domain::elements;
namespace asmm = axis::services::messaging;

aaocs::SummaryNodeStrainCollector::SummaryNodeStrainCollector( const axis::String& targetSetName, 
                                                               SummaryType summaryType ) :
SummaryNode6DCollector(targetSetName, summaryType)
{
  // nothing to do here
}

aaocs::SummaryNodeStrainCollector::SummaryNodeStrainCollector( const axis::String& targetSetName, 
                                                               SummaryType summaryType, 
                                                               aaoc::XXDirectionState xxState, 
                                                               aaoc::YYDirectionState yyState, 
                                                               aaoc::ZZDirectionState zzState, 
                                                               aaoc::YZDirectionState yzState, 
                                                               aaoc::XZDirectionState xzState, 
                                                               aaoc::XYDirectionState xyState ) :
SummaryNode6DCollector(targetSetName, summaryType, xxState, yyState, zzState, yzState, xzState, xyState)
{
  // nothing to do here
}

aaocs::SummaryNodeStrainCollector::~SummaryNodeStrainCollector( void )
{
  // nothing to do here
}

aaocs::SummaryNodeStrainCollector& aaocs::SummaryNodeStrainCollector::Create( 
  const axis::String& targetSetName, SummaryType summaryType )
{
  return *new aaocs::SummaryNodeStrainCollector(targetSetName, summaryType);
}

aaocs::SummaryNodeStrainCollector& aaocs::SummaryNodeStrainCollector::Create( 
  const axis::String& targetSetName, SummaryType summaryType, aaoc::XXDirectionState xxState, 
  aaoc::YYDirectionState yyState, aaoc::ZZDirectionState zzState, 
  aaoc::YZDirectionState yzState, aaoc::XZDirectionState xzState, aaoc::XYDirectionState xyState )
{
  return *new aaocs::SummaryNodeStrainCollector(targetSetName, summaryType, 
    xxState, yyState, zzState, yzState, xzState, xyState);
}

void aaocs::SummaryNodeStrainCollector::Destroy( void ) const
{
  delete this;
}

real aaocs::SummaryNodeStrainCollector::CollectValue( const asmm::ResultMessage&, 
                                                      const ade::Node& node, int directionIndex, 
                                                      const ada::NumericalModel& )
{
  return node.Strain()(directionIndex);
}

axis::String aaocs::SummaryNodeStrainCollector::GetVariableName( bool plural ) const
{
  if (plural) return _T("strains");
  return _T("strain");
}
