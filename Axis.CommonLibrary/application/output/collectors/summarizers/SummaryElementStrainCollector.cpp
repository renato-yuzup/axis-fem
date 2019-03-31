#include "SummaryElementStrainCollector.hpp"
#include "domain/elements/FiniteElement.hpp"
#include "domain/physics/InfinitesimalState.hpp"

namespace aaoc = axis::application::output::collectors;
namespace aaocs = aaoc::summarizers;
namespace ada = axis::domain::analyses;
namespace ade = axis::domain::elements;
namespace asmm = axis::services::messaging;

aaocs::SummaryElementStrainCollector::SummaryElementStrainCollector( const axis::String& targetSetName, 
                                                                     SummaryType summaryType ) :
SummaryElement6DCollector(targetSetName, summaryType)
{
  // nothing to do here
}

aaocs::SummaryElementStrainCollector::SummaryElementStrainCollector( const axis::String& targetSetName, 
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

aaocs::SummaryElementStrainCollector::~SummaryElementStrainCollector( void )
{
  // nothing to do here
}

aaocs::SummaryElementStrainCollector& aaocs::SummaryElementStrainCollector::Create( 
  const axis::String& targetSetName, SummaryType summaryType )
{
  return *new aaocs::SummaryElementStrainCollector(targetSetName, summaryType);
}

aaocs::SummaryElementStrainCollector& aaocs::SummaryElementStrainCollector::Create( 
  const axis::String& targetSetName, SummaryType summaryType, 
  aaoc::XXDirectionState xxState, aaoc::YYDirectionState yyState, 
  aaoc::ZZDirectionState zzState, aaoc::YZDirectionState yzState, 
  aaoc::XZDirectionState xzState, aaoc::XYDirectionState xyState )
{
  return *new aaocs::SummaryElementStrainCollector(targetSetName, summaryType, 
    xxState, yyState, zzState, yzState, xzState, xyState);
}

void aaocs::SummaryElementStrainCollector::Destroy( void ) const
{
  delete this;
}

real aaocs::SummaryElementStrainCollector::CollectValue( const asmm::ResultMessage&, 
                                                        const ade::FiniteElement& element, 
                                                        int directionIndex, 
                                                        const ada::NumericalModel& )
{
  return element.PhysicalState().Strain()(directionIndex);
}

axis::String aaocs::SummaryElementStrainCollector::GetVariableName( bool plural ) const
{
  if (plural) return _T("strains");
  return _T("strain");
}
