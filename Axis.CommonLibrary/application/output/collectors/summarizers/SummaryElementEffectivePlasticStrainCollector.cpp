#include "SummaryElementEffectivePlasticStrainCollector.hpp"
#include "domain/elements/FiniteElement.hpp"
#include "domain/physics/InfinitesimalState.hpp"

namespace aaocs = axis::application::output::collectors::summarizers;
namespace ada = axis::domain::analyses;
namespace ade = axis::domain::elements;
namespace asmm = axis::services::messaging;

aaocs::SummaryElementEffectivePlasticStrainCollector::
  SummaryElementEffectivePlasticStrainCollector(
  const axis::String& targetSetName, SummaryType summaryType ) :
  SummaryElementScalarCollector(targetSetName, summaryType)
{
  // nothing to do here
}

aaocs::SummaryElementEffectivePlasticStrainCollector::
  ~SummaryElementEffectivePlasticStrainCollector( void )
{
  // nothing to do here
}

aaocs::SummaryElementEffectivePlasticStrainCollector& 
  aaocs::SummaryElementEffectivePlasticStrainCollector::Create( 
  const axis::String& targetSetName, SummaryType summaryType )
{
  return *new SummaryElementEffectivePlasticStrainCollector(targetSetName, 
    summaryType);
}

void aaocs::SummaryElementEffectivePlasticStrainCollector::Destroy(void) const
{
  delete this;
}

axis::String 
  aaocs::SummaryElementEffectivePlasticStrainCollector::GetVariableName(void) const
{
  return _T("effective plastic strain");
}

real aaocs::SummaryElementEffectivePlasticStrainCollector::CollectValue( 
  const asmm::ResultMessage&, const ade::FiniteElement& element, 
  const ada::NumericalModel& )
{
  return element.PhysicalState().EffectivePlasticStrain();
}
