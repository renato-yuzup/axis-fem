#include "stdafx.h"
#include "TextReportNodeDisplacementCollector.hpp"
#include "domain/analyses/NumericalModel.hpp"
#include "domain/elements/Node.hpp"

namespace aaoc = axis::application::output::collectors;
namespace ada = axis::domain::analyses;
namespace ade = axis::domain::elements;
namespace afb = axis::foundation::blas;

aaoc::TextReportNodeDisplacementCollector::TextReportNodeDisplacementCollector( 
                                                        const axis::String& targetSetName, 
                                                        const bool *activeDirections ) :
TextReportNode3DCollector(targetSetName, activeDirections)
{
  // nothing to do here
}

aaoc::TextReportNodeDisplacementCollector::~TextReportNodeDisplacementCollector( void )
{
  // nothing to do here
}

void aaoc::TextReportNodeDisplacementCollector::Destroy( void ) const
{
  delete this;
}

real aaoc::TextReportNodeDisplacementCollector::GetDofData( const ada::NumericalModel& numericalModel, 
                                                            const ade::Node& node, int dofId )
{
  const afb::ColumnVector& displacement = numericalModel.Kinematics().Displacement();
  id_type componentIdx = node.GetDoF(dofId).GetId();
  return displacement(componentIdx);
}

axis::String aaoc::TextReportNodeDisplacementCollector::GetVariableName( bool plural ) const
{
  if (plural) return _T("displacements");
  return _T("displacement");
}

axis::String aaoc::TextReportNodeDisplacementCollector::GetVariableSymbol( void ) const
{
  return _T("u");
}

