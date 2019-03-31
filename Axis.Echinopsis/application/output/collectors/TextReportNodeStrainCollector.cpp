#include "stdafx.h"
#include "TextReportNodeStrainCollector.hpp"

namespace aaoc = axis::application::output::collectors;
namespace ada = axis::domain::analyses;
namespace ade = axis::domain::elements;

aaoc::TextReportNodeStrainCollector::TextReportNodeStrainCollector( const axis::String& targetSetName, 
                                                                    const bool *activeDirections ) :
TextReportNode6DCollector(targetSetName, activeDirections)
{
  // nothing to do here
}

aaoc::TextReportNodeStrainCollector::~TextReportNodeStrainCollector( void )
{
  // nothing to do here
}

void aaoc::TextReportNodeStrainCollector::Destroy( void ) const
{
  delete this;
}

real aaoc::TextReportNodeStrainCollector::GetDofData( const ada::NumericalModel& numericalModel, 
                                                      const ade::Node& node, int dofId )
{
  return node.Strain()(dofId);
}

axis::String aaoc::TextReportNodeStrainCollector::GetVariableName( bool plural ) const
{
  if (plural) return _T("strains");
  return _T("strain");
}

axis::String aaoc::TextReportNodeStrainCollector::GetVariableSymbol( void ) const
{
  return _T("e");
}
