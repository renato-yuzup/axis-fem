#include "stdafx.h"
#include "TextReportNodeVelocityCollector.hpp"
#include "domain/analyses/NumericalModel.hpp"
#include "domain/elements/Node.hpp"

namespace aaoc = axis::application::output::collectors;
namespace ada = axis::domain::analyses;
namespace ade = axis::domain::elements;
namespace afb = axis::foundation::blas;

aaoc::TextReportNodeVelocityCollector::TextReportNodeVelocityCollector( 
                                                        const axis::String& targetSetName, 
                                                        const bool *activeDirections ) :
TextReportNode3DCollector(targetSetName, activeDirections)
{
  // nothing to do here
}

aaoc::TextReportNodeVelocityCollector::~TextReportNodeVelocityCollector( void )
{
  // nothing to do here
}

void aaoc::TextReportNodeVelocityCollector::Destroy( void ) const
{
  delete this;
}

real aaoc::TextReportNodeVelocityCollector::GetDofData( const ada::NumericalModel& numericalModel, 
                                                        const ade::Node& node, int dofId )
{
  const afb::ColumnVector& velocity = numericalModel.Kinematics().Velocity();
  id_type componentIdx = node.GetDoF(dofId).GetId();
  return velocity(componentIdx);
}

axis::String aaoc::TextReportNodeVelocityCollector::GetVariableName( bool plural ) const
{
  if (plural) return _T("velocities");
  return _T("velocity");
}

axis::String aaoc::TextReportNodeVelocityCollector::GetVariableSymbol( void ) const
{
  return _T("v");
}

