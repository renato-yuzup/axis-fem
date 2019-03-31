#pragma once
#include "foundation/Axis.CommonLibrary.hpp"
#include "SummaryNode3DCollector.hpp"
#include "SummaryType.hpp"
#include "application/output/collectors/Direction3DState.hpp"

namespace axis { namespace application { namespace output { namespace collectors { namespace summarizers {

/**
 * Collects and summarizes nodal velocity data.
 *
 * @sa SummaryNode3DCollector
**/
class AXISCOMMONLIBRARY_API SummaryNodeVelocityCollector : public SummaryNode3DCollector
{
public:
  static SummaryNodeVelocityCollector& Create(const axis::String& targetSetName, SummaryType summaryType);
  static SummaryNodeVelocityCollector& Create(const axis::String& targetSetName, 
                               SummaryType summaryType,
                               axis::application::output::collectors::XDirectionState xState,
                               axis::application::output::collectors::YDirectionState yState,
                               axis::application::output::collectors::ZDirectionState zState);
  virtual ~SummaryNodeVelocityCollector(void);
  virtual void Destroy( void ) const;
private:
  SummaryNodeVelocityCollector(const axis::String& targetSetName, SummaryType summaryType);
  SummaryNodeVelocityCollector(const axis::String& targetSetName, SummaryType summaryType,
                               axis::application::output::collectors::XDirectionState xState,
                               axis::application::output::collectors::YDirectionState yState,
                               axis::application::output::collectors::ZDirectionState zState);
  virtual real CollectValue( const axis::services::messaging::ResultMessage& message, 
                             const axis::domain::elements::Node& node, int directionIndex, 
                             const axis::domain::analyses::NumericalModel& numericalModel );
  virtual axis::String GetVariableName( bool plural ) const;
};

} } } } } // namespace axis::application::output::collectors::summarizers
