#include "SummaryNodeVelocityCollector.hpp"
#include <assert.h>
#include "domain/elements/Node.hpp"
#include "domain/analyses/NumericalModel.hpp"

namespace aaocs = axis::application::output::collectors::summarizers;
namespace aaoc = axis::application::output::collectors;
namespace aaor = axis::application::output::recordsets;
namespace ada = axis::domain::analyses;
namespace ade = axis::domain::elements;
namespace asmm = axis::services::messaging;

aaocs::SummaryNodeVelocityCollector::SummaryNodeVelocityCollector(const axis::String& targetSetName, 
                                                                  SummaryType summaryType ) :
SummaryNode3DCollector(targetSetName, summaryType)
{
  // nothing to do here
}

aaocs::SummaryNodeVelocityCollector::SummaryNodeVelocityCollector(const axis::String& targetSetName, 
                                                                  SummaryType summaryType, 
                                                                  aaoc::XDirectionState xState, 
                                                                  aaoc::YDirectionState yState, 
                                                                  aaoc::ZDirectionState zState ) :
SummaryNode3DCollector(targetSetName, summaryType, xState, yState, zState)
{
  // nothing to do here
}

aaocs::SummaryNodeVelocityCollector::~SummaryNodeVelocityCollector( void )
{
  // nothing to do here
}

aaocs::SummaryNodeVelocityCollector& aaocs::SummaryNodeVelocityCollector::Create( 
  const axis::String& targetSetName, SummaryType summaryType )
{
  return *new aaocs::SummaryNodeVelocityCollector(targetSetName, summaryType);
}

aaocs::SummaryNodeVelocityCollector& aaocs::SummaryNodeVelocityCollector::Create( 
  const axis::String& targetSetName, SummaryType summaryType, aaoc::XDirectionState xState, 
  aaoc::YDirectionState yState, aaoc::ZDirectionState zState )
{
  return *new aaocs::SummaryNodeVelocityCollector(targetSetName, summaryType, xState, yState, zState);
}

void aaocs::SummaryNodeVelocityCollector::Destroy( void ) const
{
  delete this;
}

real aaocs::SummaryNodeVelocityCollector::CollectValue( const asmm::ResultMessage&, 
                                                        const ade::Node& node, int directionIndex, 
                                                        const ada::NumericalModel& numericalModel )
{
  id_type vIdx = node.GetDoF(directionIndex).GetId();
  return numericalModel.Kinematics().Velocity()(vIdx);
}

axis::String aaocs::SummaryNodeVelocityCollector::GetVariableName( bool plural ) const
{
  if (plural) return _T("velocities");
  return _T("velocity");
}
