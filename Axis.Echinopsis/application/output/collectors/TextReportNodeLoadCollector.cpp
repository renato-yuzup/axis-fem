#include "stdafx.h"
#include "TextReportNodeLoadCollector.hpp"
#include "domain/analyses/NumericalModel.hpp"
#include "domain/elements/Node.hpp"

namespace aaoc = axis::application::output::collectors;
namespace ada = axis::domain::analyses;
namespace ade = axis::domain::elements;
namespace afb = axis::foundation::blas;

aaoc::TextReportNodeLoadCollector::TextReportNodeLoadCollector( const axis::String& targetSetName, 
                                                                const bool *activeDirections ) :
TextReportNode3DCollector(targetSetName, activeDirections)
{
  // nothing to do here
}

aaoc::TextReportNodeLoadCollector::~TextReportNodeLoadCollector( void )
{
  // nothing to do here
}

void aaoc::TextReportNodeLoadCollector::Destroy( void ) const
{
  delete this;
}

real aaoc::TextReportNodeLoadCollector::GetDofData( const ada::NumericalModel& numericalModel, 
                                                    const ade::Node& node, int dofId )
{
  const afb::ColumnVector& load = numericalModel.Dynamics().ExternalLoads();
  id_type componentIdx = node.GetDoF(dofId).GetId();
  return load(componentIdx);
}

axis::String aaoc::TextReportNodeLoadCollector::GetVariableName( bool plural ) const
{
  if (plural) return _T("external loads");
  return _T("external loads");
}

axis::String aaoc::TextReportNodeLoadCollector::GetVariableSymbol( void ) const
{
  return _T("rext");
}

