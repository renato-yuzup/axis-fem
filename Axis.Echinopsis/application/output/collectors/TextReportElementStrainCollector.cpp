#include "stdafx.h"
#include "TextReportElementStrainCollector.hpp"
#include "domain/physics/InfinitesimalState.hpp"
#include "foundation/blas/blas.hpp"

namespace aaoc = axis::application::output::collectors;
namespace ada = axis::domain::analyses;
namespace ade = axis::domain::elements;

aaoc::TextReportElementStrainCollector::TextReportElementStrainCollector( 
                                                        const axis::String& targetSetName, 
                                                        const bool *activeDirections ) :
TextReportElement6DCollector(targetSetName, activeDirections)
{
  // nothing to do here
}

aaoc::TextReportElementStrainCollector::~TextReportElementStrainCollector( void )
{
  // nothing to do here
}

void aaoc::TextReportElementStrainCollector::Destroy( void ) const
{
  delete this;
}

real aaoc::TextReportElementStrainCollector::GetDofData( const ada::NumericalModel& numericalModel, 
                                                         const ade::FiniteElement& element, 
                                                         int directionIdx )
{
  return element.PhysicalState().Strain()(directionIdx);
}

axis::String aaoc::TextReportElementStrainCollector::GetVariableName( bool plural ) const
{
  if (plural) return _T("strains");
  return _T("strain");
}

axis::String aaoc::TextReportElementStrainCollector::GetVariableSymbol( void ) const
{
  return _T("e");
}
