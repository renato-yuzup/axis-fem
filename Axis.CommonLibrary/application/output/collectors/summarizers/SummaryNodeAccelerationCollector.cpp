#include "SummaryNodeAccelerationCollector.hpp"
#include <assert.h>
#include "domain/elements/Node.hpp"
#include "domain/analyses/NumericalModel.hpp"

namespace aaocs = axis::application::output::collectors::summarizers;
namespace aaoc = axis::application::output::collectors;
namespace aaor = axis::application::output::recordsets;
namespace ada = axis::domain::analyses;
namespace ade = axis::domain::elements;
namespace asmm = axis::services::messaging;

aaocs::SummaryNodeAccelerationCollector::SummaryNodeAccelerationCollector( 
  const axis::String& targetSetName, 
  SummaryType summaryType ) :
SummaryNode3DCollector(targetSetName, summaryType)
{
  // nothing to do here
}

aaocs::SummaryNodeAccelerationCollector::SummaryNodeAccelerationCollector( 
  const axis::String& targetSetName, 
  SummaryType summaryType, 
  aaoc::XDirectionState xState, 
  aaoc::YDirectionState yState, 
  aaoc::ZDirectionState zState ) :
SummaryNode3DCollector(targetSetName, summaryType, xState, yState, zState)
{
  // nothing to do here
}

aaocs::SummaryNodeAccelerationCollector::~SummaryNodeAccelerationCollector( void )
{
  // nothing to do here
}

aaocs::SummaryNodeAccelerationCollector& aaocs::SummaryNodeAccelerationCollector::Create( 
  const axis::String& targetSetName, SummaryType summaryType )
{
  return *new aaocs::SummaryNodeAccelerationCollector(targetSetName, summaryType);
}

aaocs::SummaryNodeAccelerationCollector& aaocs::SummaryNodeAccelerationCollector::Create( 
  const axis::String& targetSetName, SummaryType summaryType, aaoc::XDirectionState xState, 
  aaoc::YDirectionState yState, aaoc::ZDirectionState zState )
{
  return *new aaocs::SummaryNodeAccelerationCollector(targetSetName, summaryType, xState, yState, zState);
}

void aaocs::SummaryNodeAccelerationCollector::Destroy( void ) const
{
  delete this;
}

real aaocs::SummaryNodeAccelerationCollector::CollectValue( const asmm::ResultMessage&, 
                                                            const ade::Node& node, int directionIndex, 
                                                            const ada::NumericalModel& numericalModel )
{
  id_type aIdx = node.GetDoF(directionIndex).GetId();
  return numericalModel.Kinematics().Acceleration()(aIdx);
}

axis::String aaocs::SummaryNodeAccelerationCollector::GetVariableName( bool plural ) const
{
  if (plural) return _T("accelerations");
  return _T("acceleration");
}
