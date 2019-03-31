#include "SummaryElementPlasticStrainCollector.hpp"
#include "domain/elements/FiniteElement.hpp"
#include "domain/physics/InfinitesimalState.hpp"

namespace aaoc = axis::application::output::collectors;
namespace aaocs = aaoc::summarizers;
namespace ada = axis::domain::analyses;
namespace ade = axis::domain::elements;
namespace asmm = axis::services::messaging;

aaocs::SummaryElementPlasticStrainCollector::
  SummaryElementPlasticStrainCollector( 
  const axis::String& targetSetName, SummaryType summaryType ) :
  SummaryElement6DCollector(targetSetName, summaryType)
{
  // nothing to do here
}

aaocs::SummaryElementPlasticStrainCollector::
  SummaryElementPlasticStrainCollector( 
  const axis::String& targetSetName, SummaryType summaryType, 
  aaoc::XXDirectionState xxState, aaoc::YYDirectionState yyState, 
  aaoc::ZZDirectionState zzState, aaoc::YZDirectionState yzState, 
  aaoc::XZDirectionState xzState, aaoc::XYDirectionState xyState ) :
  SummaryElement6DCollector(targetSetName, summaryType, xxState, yyState, 
  zzState, yzState, xzState, xyState)
{
  // nothing to do here
}

aaocs::SummaryElementPlasticStrainCollector::
  ~SummaryElementPlasticStrainCollector( void )
{
  // nothing to do here
}

aaocs::SummaryElementPlasticStrainCollector& 
  aaocs::SummaryElementPlasticStrainCollector::Create( 
  const axis::String& targetSetName, SummaryType summaryType )
{
  return *new aaocs::SummaryElementPlasticStrainCollector(
    targetSetName, summaryType);
}

aaocs::SummaryElementPlasticStrainCollector& 
  aaocs::SummaryElementPlasticStrainCollector::Create( 
  const axis::String& targetSetName, SummaryType summaryType, 
  aaoc::XXDirectionState xxState, aaoc::YYDirectionState yyState, 
  aaoc::ZZDirectionState zzState, aaoc::YZDirectionState yzState, 
  aaoc::XZDirectionState xzState, aaoc::XYDirectionState xyState )
{
  return *new aaocs::SummaryElementPlasticStrainCollector(
    targetSetName, summaryType, xxState, yyState, zzState, yzState, 
    xzState, xyState);
}

void aaocs::SummaryElementPlasticStrainCollector::Destroy(void) const
{
  delete this;
}

real aaocs::SummaryElementPlasticStrainCollector::CollectValue( 
  const asmm::ResultMessage&, const ade::FiniteElement& element, 
  int directionIndex, const ada::NumericalModel& )
{
  return element.PhysicalState().PlasticStrain()(directionIndex);
}

axis::String 
  aaocs::SummaryElementPlasticStrainCollector::GetVariableName( 
  bool plural ) const
{
  if (plural) return _T("plastic strain");
  return _T("plastic strain");
}
