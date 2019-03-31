#include "SummaryNodeExternalLoadCollector.hpp"
#include <assert.h>
#include "domain/elements/Node.hpp"
#include "domain/analyses/NumericalModel.hpp"

namespace aaocs = axis::application::output::collectors::summarizers;
namespace aaoc = axis::application::output::collectors;
namespace aaor = axis::application::output::recordsets;
namespace ada = axis::domain::analyses;
namespace ade = axis::domain::elements;
namespace asmm = axis::services::messaging;
namespace afb = axis::foundation::blas;

aaocs::SummaryNodeExternalLoadCollector::SummaryNodeExternalLoadCollector( 
  const axis::String& targetSetName, 
  SummaryType summaryType ) :
SummaryNode3DCollector(targetSetName, summaryType)
{
  // nothing to do here
}

aaocs::SummaryNodeExternalLoadCollector::SummaryNodeExternalLoadCollector( 
  const axis::String& targetSetName, 
  SummaryType summaryType, 
  aaoc::XDirectionState xState, 
  aaoc::YDirectionState yState, 
  aaoc::ZDirectionState zState ) :
SummaryNode3DCollector(targetSetName, summaryType, xState, yState, zState)
{
  // nothing to do here
}

aaocs::SummaryNodeExternalLoadCollector::~SummaryNodeExternalLoadCollector( void )
{
  // nothing to do here
}

aaocs::SummaryNodeExternalLoadCollector& aaocs::SummaryNodeExternalLoadCollector::Create( 
  const axis::String& targetSetName, SummaryType summaryType )
{
  return *new aaocs::SummaryNodeExternalLoadCollector(targetSetName, summaryType);
}

aaocs::SummaryNodeExternalLoadCollector& aaocs::SummaryNodeExternalLoadCollector::Create( 
  const axis::String& targetSetName, SummaryType summaryType, aaoc::XDirectionState xState, 
  aaoc::YDirectionState yState, aaoc::ZDirectionState zState )
{
  return *new aaocs::SummaryNodeExternalLoadCollector(targetSetName, summaryType, xState, yState, zState);
}

void aaocs::SummaryNodeExternalLoadCollector::Destroy( void ) const
{
  delete this;
}

real aaocs::SummaryNodeExternalLoadCollector::CollectValue( const asmm::ResultMessage&, 
                                                            const ade::Node& node, int directionIndex, 
                                                            const ada::NumericalModel& numericalModel )
{
  id_type loadIdx = node.GetDoF(directionIndex).GetId();
  const afb::ColumnVector& loads = numericalModel.Dynamics().ExternalLoads();
  return loads(loadIdx);
}

axis::String aaocs::SummaryNodeExternalLoadCollector::GetVariableName( bool plural ) const
{
  if (plural) return _T("external loads");
  return _T("external load");
}
