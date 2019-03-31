#pragma once
#include "foundation/Axis.CommonLibrary.hpp"
#include "SummaryElement6DCollector.hpp"
#include "SummaryType.hpp"
#include "application/output/collectors/Direction6DState.hpp"

namespace axis { namespace application { namespace output { namespace collectors { namespace summarizers {

/**
 * Collects and summarizes element strain data.
 *
 * @sa SummaryElement6DCollector
**/
class AXISCOMMONLIBRARY_API SummaryElementStrainCollector : public SummaryElement6DCollector
{
public:
  static SummaryElementStrainCollector& Create(const axis::String& targetSetName, SummaryType summaryType);
  static SummaryElementStrainCollector& Create(const axis::String& targetSetName, SummaryType summaryType,
                                axis::application::output::collectors::XXDirectionState xxState,
                                axis::application::output::collectors::YYDirectionState yyState,
                                axis::application::output::collectors::ZZDirectionState zzState,
                                axis::application::output::collectors::YZDirectionState yzState,
                                axis::application::output::collectors::XZDirectionState xzState,
                                axis::application::output::collectors::XYDirectionState xyState);
  virtual ~SummaryElementStrainCollector(void);
  virtual void Destroy( void ) const;
private:
  SummaryElementStrainCollector(const axis::String& targetSetName, SummaryType summaryType);
  SummaryElementStrainCollector(const axis::String& targetSetName, SummaryType summaryType,
                                axis::application::output::collectors::XXDirectionState xxState,
                                axis::application::output::collectors::YYDirectionState yyState,
                                axis::application::output::collectors::ZZDirectionState zzState,
                                axis::application::output::collectors::YZDirectionState yzState,
                                axis::application::output::collectors::XZDirectionState xzState,
                                axis::application::output::collectors::XYDirectionState xyState);
  virtual axis::String GetVariableName( bool plural ) const;
  virtual real CollectValue( const axis::services::messaging::ResultMessage& message, 
                             const axis::domain::elements::FiniteElement& element, int directionIndex, 
                             const axis::domain::analyses::NumericalModel& numericalModel );
};

} } } } } // namespace axis::application::output::collectors::summarizers