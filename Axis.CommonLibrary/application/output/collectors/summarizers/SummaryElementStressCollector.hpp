#pragma once
#include "foundation/Axis.CommonLibrary.hpp"
#include "SummaryElement6DCollector.hpp"
#include "SummaryType.hpp"
#include "application/output/collectors/Direction6DState.hpp"

namespace axis { namespace application { namespace output { namespace collectors { namespace summarizers {

/**
 * Collects and summarizes element stress data.
 *
 * @sa SummaryElement6DCollector
**/
class AXISCOMMONLIBRARY_API SummaryElementStressCollector : public SummaryElement6DCollector
{
public:
  static SummaryElementStressCollector& Create(const axis::String& targetSetName, SummaryType summaryType);
  static SummaryElementStressCollector& Create(const axis::String& targetSetName, SummaryType summaryType,
                                axis::application::output::collectors::XXDirectionState xxState,
                                axis::application::output::collectors::YYDirectionState yyState,
                                axis::application::output::collectors::ZZDirectionState zzState,
                                axis::application::output::collectors::YZDirectionState yzState,
                                axis::application::output::collectors::XZDirectionState xzState,
                                axis::application::output::collectors::XYDirectionState xyState);
  virtual ~SummaryElementStressCollector(void);
  virtual void Destroy( void ) const;
private:
  SummaryElementStressCollector(const axis::String& targetSetName, SummaryType summaryType);
  SummaryElementStressCollector(const axis::String& targetSetName, SummaryType summaryType,
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
