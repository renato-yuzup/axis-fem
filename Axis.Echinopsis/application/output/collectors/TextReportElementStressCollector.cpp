#include "stdafx.h"
#include "TextReportElementStressCollector.hpp"
#include "domain/physics/InfinitesimalState.hpp"

namespace aaoc = axis::application::output::collectors;
namespace ada = axis::domain::analyses;
namespace ade = axis::domain::elements;

aaoc::TextReportElementStressCollector::TextReportElementStressCollector( 
                                                        const axis::String& targetSetName, 
                                                        const bool *activeDirections ) :
TextReportElement6DCollector(targetSetName, activeDirections)
{
  // nothing to do here
}

aaoc::TextReportElementStressCollector::~TextReportElementStressCollector( void )
{
  // nothing to do here
}

void aaoc::TextReportElementStressCollector::Destroy( void ) const
{
  delete this;
}

real aaoc::TextReportElementStressCollector::GetDofData( const ada::NumericalModel& numericalModel, 
                                                         const ade::FiniteElement& element, 
                                                         int directionIdx )
{
  return element.PhysicalState().Stress()(directionIdx);
}

axis::String aaoc::TextReportElementStressCollector::GetVariableName( bool plural ) const
{
  if (plural) return _T("stresses");
  return _T("stress");
}

axis::String aaoc::TextReportElementStressCollector::GetVariableSymbol( void ) const
{
  return _T("s");
}
