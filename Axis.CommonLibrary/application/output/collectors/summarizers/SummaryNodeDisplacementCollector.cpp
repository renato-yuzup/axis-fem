#include "SummaryNodeDisplacementCollector.hpp"
#include <assert.h>
#include "domain/elements/Node.hpp"
#include "domain/analyses/NumericalModel.hpp"

namespace aaocs = axis::application::output::collectors::summarizers;
namespace aaoc = axis::application::output::collectors;
namespace aaor = axis::application::output::recordsets;
namespace ada = axis::domain::analyses;
namespace ade = axis::domain::elements;
namespace asmm = axis::services::messaging;

aaocs::SummaryNodeDisplacementCollector::SummaryNodeDisplacementCollector( 
                                                        const axis::String& targetSetName, 
                                                        SummaryType summaryType ) :
SummaryNode3DCollector(targetSetName, summaryType)
{
  // nothing to do here
}

aaocs::SummaryNodeDisplacementCollector::SummaryNodeDisplacementCollector( 
                                                        const axis::String& targetSetName, 
                                                        SummaryType summaryType, 
                                                        aaoc::XDirectionState xState, 
                                                        aaoc::YDirectionState yState, 
                                                        aaoc::ZDirectionState zState ) :
SummaryNode3DCollector(targetSetName, summaryType, xState, yState, zState)
{
  // nothing to do here
}

aaocs::SummaryNodeDisplacementCollector::~SummaryNodeDisplacementCollector( void )
{
  // nothing to do here
}

aaocs::SummaryNodeDisplacementCollector& aaocs::SummaryNodeDisplacementCollector::Create( 
  const axis::String& targetSetName, SummaryType summaryType )
{
  return *new aaocs::SummaryNodeDisplacementCollector(targetSetName, summaryType);
}

aaocs::SummaryNodeDisplacementCollector& aaocs::SummaryNodeDisplacementCollector::Create( 
  const axis::String& targetSetName, SummaryType summaryType, aaoc::XDirectionState xState, 
  aaoc::YDirectionState yState, aaoc::ZDirectionState zState )
{
  return *new aaocs::SummaryNodeDisplacementCollector(targetSetName, summaryType, xState, yState, zState);
}

void aaocs::SummaryNodeDisplacementCollector::Destroy( void ) const
{
  delete this;
}

real aaocs::SummaryNodeDisplacementCollector::CollectValue( const asmm::ResultMessage&, 
                                                            const ade::Node& node, int directionIndex, 
                                                            const ada::NumericalModel& numericalModel )
{
  id_type dispIdx = node.GetDoF(directionIndex).GetId();
  return numericalModel.Kinematics().Displacement()(dispIdx);
}

axis::String aaocs::SummaryNodeDisplacementCollector::GetVariableName( bool plural ) const
{
  if (plural) return _T("displacements");
  return _T("displacement");
}
