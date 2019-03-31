#include "SummaryNodeReactionForceCollector.hpp"
#include <assert.h>
#include "domain/elements/Node.hpp"
#include "domain/analyses/NumericalModel.hpp"

namespace aaocs = axis::application::output::collectors::summarizers;
namespace aaoc = axis::application::output::collectors;
namespace aaor = axis::application::output::recordsets;
namespace ada = axis::domain::analyses;
namespace ade = axis::domain::elements;
namespace asmm = axis::services::messaging;

aaocs::SummaryNodeReactionForceCollector::SummaryNodeReactionForceCollector( 
                                                        const axis::String& targetSetName, 
                                                        SummaryType summaryType ) :
SummaryNode3DCollector(targetSetName, summaryType)
{
  // nothing to do here
}

aaocs::SummaryNodeReactionForceCollector::SummaryNodeReactionForceCollector( 
                                                        const axis::String& targetSetName, 
                                                        SummaryType summaryType, 
                                                        aaoc::XDirectionState xState, 
                                                        aaoc::YDirectionState yState, 
                                                        aaoc::ZDirectionState zState ) :
SummaryNode3DCollector(targetSetName, summaryType, xState, yState, zState)
{
  // nothing to do here
}

aaocs::SummaryNodeReactionForceCollector::~SummaryNodeReactionForceCollector( void )
{
  // nothing to do here
}

aaocs::SummaryNodeReactionForceCollector& aaocs::SummaryNodeReactionForceCollector::Create( 
  const axis::String& targetSetName, SummaryType summaryType )
{
  return *new aaocs::SummaryNodeReactionForceCollector(targetSetName, summaryType);
}

aaocs::SummaryNodeReactionForceCollector& aaocs::SummaryNodeReactionForceCollector::Create( 
  const axis::String& targetSetName, SummaryType summaryType, aaoc::XDirectionState xState, 
  aaoc::YDirectionState yState, aaoc::ZDirectionState zState )
{
  return *new aaocs::SummaryNodeReactionForceCollector(targetSetName, summaryType, xState, yState, zState);
}

void aaocs::SummaryNodeReactionForceCollector::Destroy( void ) const
{
  delete this;
}

real aaocs::SummaryNodeReactionForceCollector::CollectValue( const asmm::ResultMessage&, 
                                                             const ade::Node& node, int directionIndex, 
                                                             const ada::NumericalModel& numericalModel )
{
  id_type rfIdx = node.GetDoF(directionIndex).GetId();
  return numericalModel.Dynamics().InternalForces()(rfIdx);
}

axis::String aaocs::SummaryNodeReactionForceCollector::GetVariableName( bool plural ) const
{
  if (plural) return _T("reaction forces");
  return _T("reaction force");
}
