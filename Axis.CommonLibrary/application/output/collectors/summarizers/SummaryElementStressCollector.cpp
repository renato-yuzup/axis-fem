#include "SummaryElementStressCollector.hpp"
#include "domain/elements/FiniteElement.hpp"
#include "domain/physics/InfinitesimalState.hpp"

namespace aaoc = axis::application::output::collectors;
namespace aaocs = aaoc::summarizers;
namespace ada = axis::domain::analyses;
namespace ade = axis::domain::elements;
namespace asmm = axis::services::messaging;

aaocs::SummaryElementStressCollector::SummaryElementStressCollector( const axis::String& targetSetName, 
                                                                     SummaryType summaryType ) :
SummaryElement6DCollector(targetSetName, summaryType)
{
  // nothing to do here
}

aaocs::SummaryElementStressCollector::SummaryElementStressCollector( const axis::String& targetSetName, 
                                                                     SummaryType summaryType, 
                                                                     aaoc::XXDirectionState xxState, 
                                                                     aaoc::YYDirectionState yyState, 
                                                                     aaoc::ZZDirectionState zzState, 
                                                                     aaoc::YZDirectionState yzState, 
                                                                     aaoc::XZDirectionState xzState, 
                                                                     aaoc::XYDirectionState xyState ) :
SummaryElement6DCollector(targetSetName, summaryType, xxState, yyState, zzState, yzState, xzState, xyState)
{
  // nothing to do here
}

aaocs::SummaryElementStressCollector::~SummaryElementStressCollector( void )
{
  // nothing to do here
}

aaocs::SummaryElementStressCollector& aaocs::SummaryElementStressCollector::Create( 
  const axis::String& targetSetName, SummaryType summaryType )
{
  return *new aaocs::SummaryElementStressCollector(targetSetName, summaryType);
}

aaocs::SummaryElementStressCollector& aaocs::SummaryElementStressCollector::Create( 
  const axis::String& targetSetName, SummaryType summaryType, aaoc::XXDirectionState xxState, 
  aaoc::YYDirectionState yyState, aaoc::ZZDirectionState zzState, 
  aaoc::YZDirectionState yzState, aaoc::XZDirectionState xzState, aaoc::XYDirectionState xyState )
{
  return *new aaocs::SummaryElementStressCollector(targetSetName, summaryType, xxState, yyState, 
                                                   zzState, yzState, xzState, xyState);
}

void aaocs::SummaryElementStressCollector::Destroy( void ) const
{
  delete this;
}

real aaocs::SummaryElementStressCollector::CollectValue( const asmm::ResultMessage&, 
                                                         const ade::FiniteElement& element, 
                                                         int directionIndex, 
                                                         const ada::NumericalModel& )
{
  return element.PhysicalState().Stress()(directionIndex);
}

axis::String aaocs::SummaryElementStressCollector::GetVariableName( bool plural ) const
{
  if (plural) return _T("stresses");
  return _T("stress");
}
