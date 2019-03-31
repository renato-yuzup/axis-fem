#include "stdafx.h"
#include "LinearSimpleHexahedronFactory.hpp"
#include "AxisString.hpp"
#include "application/parsing/core/SectionDefinition.hpp"
#include "domain/elements/ElementGeometry.hpp"
#include "foundation/ArgumentException.hpp"
#include "domain/formulations/LinearHexaReducedFormulation.hpp"
#include "domain/formulations/LinearHexaFullFormulation.hpp"
#include "Domain/Integration/IntegrationPoint.hpp"

namespace aapc = axis::application::parsing::core;
namespace aafe = axis::application::factories::elements;
namespace ada = axis::domain::analyses;
namespace ade = axis::domain::elements;
namespace adi = axis::domain::integration;
namespace adf = axis::domain::formulations;

aafe::LinearSimpleHexahedronFactory::LinearSimpleHexahedronFactory( void )
{
  // nothing to do here
}

aafe::LinearSimpleHexahedronFactory::~LinearSimpleHexahedronFactory( void )
{
  // nothing to do here
}

bool aafe::LinearSimpleHexahedronFactory::IsValidDefinition( const aapc::SectionDefinition& definition )
{
  bool canBuild = true;
  int propertiesDefined = 0;

  if (definition.GetSectionTypeName() != _T("LINEAR_HEXAHEDRON")) return false;

  if (definition.IsPropertyDefined(_T("INTEGRATION_TYPE")))
  {
    propertiesDefined++;
    // check if the supplied value is valid
    axis::String value = definition.GetPropertyValue(_T("INTEGRATION_TYPE")).ToString();
    if (!(value.equals(_T("REDUCED")) || value.equals(_T("FULL")))) return false;
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

adf::Formulation& aafe::LinearSimpleHexahedronFactory::BuildFormulation( 
  const aapc::SectionDefinition& sectionDefinition, axis::domain::elements::ElementGeometry& geometry )
{
  if (geometry.GetIntegrationPointCount() == 1)  // reduced integration
  {
    return *new adf::LinearHexaReducedFormulation();
  }

  // full integration
  return *new adf::LinearHexaFullFormulation();
}

void aafe::LinearSimpleHexahedronFactory::BuildIntegrationPoint( ade::ElementGeometry& geometry, 
  const aapc::SectionDefinition& sectionDefinition )
{
  // get how many integration points the user wants
  String integrationTypeOption;
  if (sectionDefinition.IsPropertyDefined(_T("INTEGRATION_TYPE")))
  {
    integrationTypeOption = sectionDefinition.GetPropertyValue(_T("INTEGRATION_TYPE")).ToString();
  }
  else
  {
    integrationTypeOption = _T("FULL");
  }

  // determine integration type
  if (integrationTypeOption.equals(_T("REDUCED")))	// 1-point numerical integration
  {
    geometry.SetIntegrationPoint(0, adi::IntegrationPoint::Create(0, 0, 0, 8));
  }
  else if(integrationTypeOption.equals(_T("FULL")))	// 8-point numerical integration
  {
    const coordtype x = (coordtype)0.57735026918962576450914878050196;
    int idx = 0;
    for (int i = -1; i <= 1; i += 2)
    {
      for (int j = -1; j <= 1; j += 2)
      {
        for (int k = -1; k <= 1; k += 2)
        {
          geometry.SetIntegrationPoint(idx, adi::IntegrationPoint::Create(i*x, j*x, k*x, 1.0));
        }
      }
    }		
  }
  else
  {	// option not recognized
    throw axis::foundation::ArgumentException();
  }
}

int aafe::LinearSimpleHexahedronFactory::GetIntegrationPointCount( 
  const aapc::SectionDefinition& sectionDefinition ) const
{
  // get how many integration points the user wants
  String integrationTypeOption;
  if (sectionDefinition.IsPropertyDefined(_T("INTEGRATION_TYPE")))
  {
    integrationTypeOption = sectionDefinition.GetPropertyValue(_T("INTEGRATION_TYPE")).ToString();
  }
  else
  {
    integrationTypeOption = _T("FULL");
  }

  if (integrationTypeOption.equals(_T("REDUCED")))	// 1-point numerical integration
  {
    return 1;
  }
  else if(integrationTypeOption.equals(_T("FULL")))	// 8-point numerical integration
  {
    return 8;
  }
  else
  {	// option not recognized
    throw axis::foundation::ArgumentException();
  }
}
