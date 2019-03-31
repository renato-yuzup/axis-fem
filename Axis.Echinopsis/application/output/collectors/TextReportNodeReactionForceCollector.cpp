#include "stdafx.h"
#include "TextReportNodeReactionForceCollector.hpp"
#include "domain/analyses/NumericalModel.hpp"
#include "domain/elements/Node.hpp"

namespace aaoc = axis::application::output::collectors;
namespace ada = axis::domain::analyses;
namespace ade = axis::domain::elements;
namespace afb = axis::foundation::blas;

aaoc::TextReportNodeReactionForceCollector::TextReportNodeReactionForceCollector( 
                                                        const axis::String& targetSetName, 
                                                        const bool *activeDirections ) :
TextReportNode3DCollector(targetSetName, activeDirections)
{
  // nothing to do here
}

aaoc::TextReportNodeReactionForceCollector::~TextReportNodeReactionForceCollector( void )
{
  // nothing to do here
}

void aaoc::TextReportNodeReactionForceCollector::Destroy( void ) const
{
  delete this;
}

real aaoc::TextReportNodeReactionForceCollector::GetDofData( const ada::NumericalModel& numericalModel, 
                                                             const ade::Node& node, int dofId )
{
  const afb::ColumnVector& reaction = numericalModel.Dynamics().InternalForces();
  id_type componentIdx = node.GetDoF(dofId).GetId();
  return reaction(componentIdx);
}

axis::String aaoc::TextReportNodeReactionForceCollector::GetVariableName( bool plural ) const
{
  if (plural) return _T("reaction forces");
  return _T("reaction force");
}

axis::String aaoc::TextReportNodeReactionForceCollector::GetVariableSymbol( void ) const
{
  return _T("rint");
}

