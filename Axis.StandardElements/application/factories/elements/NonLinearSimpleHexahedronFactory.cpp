#include "stdafx.h"
#include "NonLinearSimpleHexahedronFactory.hpp"
#include "AxisString.hpp"
#include "application/parsing/core/SectionDefinition.hpp"
#include "domain/elements/ElementGeometry.hpp"
#include "foundation/ArgumentException.hpp"
#include "domain/formulations/NonLinearHexaReducedFormulation.hpp"
#include "Domain/Integration/IntegrationPoint.hpp"

namespace aapc = axis::application::parsing::core;
namespace aafe = axis::application::factories::elements;
namespace ada = axis::domain::analyses;
namespace ade = axis::domain::elements;
namespace adi = axis::domain::integration;
namespace adf = axis::domain::formulations;

aafe::NonLinearSimpleHexahedronFactory::NonLinearSimpleHexahedronFactory( void )
{
  // nothing to do here
}

aafe::NonLinearSimpleHexahedronFactory::~NonLinearSimpleHexahedronFactory( void )
{
  // nothing to do here
}

bool aafe::NonLinearSimpleHexahedronFactory::IsValidDefinition( const aapc::SectionDefinition& definition )
{
  bool canBuild = true;
  int propertiesDefined = 0;

  if (definition.GetSectionTypeName() != _T("NONLINEAR_HEXAHEDRON") &&
      definition.GetSectionTypeName() != _T("HEXAHEDRON")) return false;

  if (definition.IsPropertyDefined(_T("INTEGRATION_TYPE")))
  {
    propertiesDefined++;
    // check if the supplied value is valid
    axis::String value = definition.GetPropertyValue(_T("INTEGRATION_TYPE")).ToString();
    if (!value.equals(_T("REDUCED"))) return false;
  }
  if (definition.IsPropertyDefined(_T("HOURGLASS_CONTROL")))
  {
    propertiesDefined++;
    // check if the supplied value is valid
    axis::String value = definition.GetPropertyValue(_T("HOURGLASS_CONTROL")).ToString().to_upper_case();
    if (!(value.equals(_T("NO")) || value.equals(_T("FALSE")) || value.equals(_T("OFF")))) return false;
  }

  // check if only known properties and flags were supplied
  if (definition.PropertyCount() != propertiesDefined || definition.FlagCount() > 0) return false;

  // everything is ok
  return true;
}

adf::Formulation& aafe::NonLinearSimpleHexahedronFactory::BuildFormulation( 
  const aapc::SectionDefinition& sectionDefinition, axis::domain::elements::ElementGeometry& geometry )
{
  return *new adf::NonLinearHexaReducedFormulation();
}

void aafe::NonLinearSimpleHexahedronFactory::BuildIntegrationPoint( ade::ElementGeometry& geometry, 
  const aapc::SectionDefinition& sectionDefinition )
{
  geometry.SetIntegrationPoint(0, adi::IntegrationPoint::Create(0, 0, 0, 8));
}

int aafe::NonLinearSimpleHexahedronFactory::GetIntegrationPointCount( 
  const aapc::SectionDefinition& ) const
{
  return 1;
}
