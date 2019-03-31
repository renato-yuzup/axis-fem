#include "stdafx.h"
#include "TextReportNodeStressCollector.hpp"

namespace aaoc = axis::application::output::collectors;
namespace ada = axis::domain::analyses;
namespace ade = axis::domain::elements;

aaoc::TextReportNodeStressCollector::TextReportNodeStressCollector( const axis::String& targetSetName, 
                                                                    const bool *activeDirections ) :
TextReportNode6DCollector(targetSetName, activeDirections)
{
  // nothing to do here
}

aaoc::TextReportNodeStressCollector::~TextReportNodeStressCollector( void )
{
  // nothing to do here
}

void aaoc::TextReportNodeStressCollector::Destroy( void ) const
{
  delete this;
}

real aaoc::TextReportNodeStressCollector::GetDofData( const ada::NumericalModel& numericalModel, 
                                                      const ade::Node& node, int dofId )
{
  return node.Stress()(dofId);
}

axis::String aaoc::TextReportNodeStressCollector::GetVariableName( bool plural ) const
{
  if (plural) return _T("stresses");
  return _T("stress");
}

axis::String aaoc::TextReportNodeStressCollector::GetVariableSymbol( void ) const
{
  return _T("s");
}
