#pragma once
#include "foundation/Axis.CommonLibrary.hpp"
#include "SummaryElementScalarCollector.hpp"
#include "SummaryType.hpp"

namespace axis { namespace application { namespace output { namespace collectors { namespace summarizers {

/**
 * Represents a collector that summarizes scalar data from elements.
 *
 * @sa SummaryElementScalarCollector
**/
class AXISCOMMONLIBRARY_API SummaryElementEffectivePlasticStrainCollector : public SummaryElementScalarCollector
{
public:
  static SummaryElementEffectivePlasticStrainCollector& Create(const axis::String& targetSetName, SummaryType summaryType);
  virtual ~SummaryElementEffectivePlasticStrainCollector(void);
  virtual void Destroy( void ) const;
private:
  SummaryElementEffectivePlasticStrainCollector(const axis::String& targetSetName, SummaryType summaryType);
  virtual axis::String GetVariableName(void) const;
  virtual real CollectValue(const axis::services::messaging::ResultMessage& message, 
                            const axis::domain::elements::FiniteElement& element,
                            const axis::domain::analyses::NumericalModel& numericalModel);
};

} } } } } // namespace axis::application::output::collectors::summarizers
