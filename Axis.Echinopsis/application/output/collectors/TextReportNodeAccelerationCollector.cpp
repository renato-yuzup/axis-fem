#include "stdafx.h"
#include "TextReportNodeAccelerationCollector.hpp"
#include "domain/analyses/NumericalModel.hpp"
#include "domain/elements/Node.hpp"

namespace aaoc = axis::application::output::collectors;
namespace ada = axis::domain::analyses;
namespace ade = axis::domain::elements;
namespace afb = axis::foundation::blas;

aaoc::TextReportNodeAccelerationCollector::TextReportNodeAccelerationCollector( 
                                                        const axis::String& targetSetName, 
                                                        const bool *activeDirections ) :
TextReportNode3DCollector(targetSetName, activeDirections)
{
  // nothing to do here
}

aaoc::TextReportNodeAccelerationCollector::~TextReportNodeAccelerationCollector( void )
{
  // nothing to do here
}

void aaoc::TextReportNodeAccelerationCollector::Destroy( void ) const
{
  delete this;
}

real aaoc::TextReportNodeAccelerationCollector::GetDofData( const ada::NumericalModel& numericalModel, 
                                                            const ade::Node& node, int dofId )
{
  const afb::ColumnVector& acceleration = numericalModel.Kinematics().Acceleration();
  id_type componentIdx = node.GetDoF(dofId).GetId();
  return acceleration(componentIdx);
}

axis::String aaoc::TextReportNodeAccelerationCollector::GetVariableName( bool plural ) const
{
  if (plural) return _T("accelerations");
  return _T("acceleration");
}

axis::String aaoc::TextReportNodeAccelerationCollector::GetVariableSymbol( void ) const
{
  return _T("a");
}

