#include "stdafx.h"
#include "LinearFlanaganBelytschkoHexaFactory.hpp"
#include "AxisString.hpp"
#include "application/parsing/core/SectionDefinition.hpp"
#include "domain/elements/ElementGeometry.hpp"
#include "domain/formulations/LinearHexaFlanaganBelytschkoFormulation.hpp"
#include "services/language/syntax/evaluation/ParameterSyntax.hpp"
#include "foundation/ArgumentException.hpp"

namespace aapc = axis::application::parsing::core;
namespace aafe = axis::application::factories::elements;
namespace ada = axis::domain::analyses;
namespace ade = axis::domain::elements;
namespace adi = axis::domain::integration;
namespace adf = axis::domain::formulations;
namespace aslse = axis::services::language::syntax::evaluation;

aafe::LinearFlanaganBelytschkoHexaFactory::LinearFlanaganBelytschkoHexaFactory( void )
{
  // nothing to do here
}

aafe::LinearFlanaganBelytschkoHexaFactory::~LinearFlanaganBelytschkoHexaFactory( void )
{
  // nothing to do here
}

bool aafe::LinearFlanaganBelytschkoHexaFactory::IsValidDefinition( 
  const aapc::SectionDefinition& definition )
{
  bool canBuild = true;
  int propertiesDefined = 1;

  if (definition.GetSectionTypeName() != _T("LINEAR_HEXAHEDRON")) return false;

  if (!definition.IsPropertyDefined(_T("HOURGLASS_CONTROL"))) return false;
  aslse::ParameterValue& value = definition.GetPropertyValue(_T("HOURGLASS_CONTROL"));
  if (!aslse::ParameterSyntax::ToBoolean(value)) return false;

  if (definition.IsPropertyDefined(_T("HOURGLASS_CONTROL_TYPE")))
  {
    propertiesDefined++;
    // check if the supplied value is valid
    axis::String value = definition.GetPropertyValue(_T("HOURGLASS_CONTROL_TYPE")).ToString();
    if (value != _T("VISCOUS")) return false;
  }
  if (definition.IsPropertyDefined(_T("STABILIZATION_COEFFICIENT")))
  {
    propertiesDefined++;
    // check if the supplied value is valid
    aslse::ParameterValue& paramVal = definition.GetPropertyValue(_T("STABILIZATION_COEFFICIENT"));
    if (!aslse::ParameterSyntax::IsNumeric(paramVal)) return false;
  }

  // check if only known properties and flags were supplied
  if (definition.PropertyCount() != propertiesDefined || definition.FlagCount() > 0) return false;

  // everything is ok
  return true;
}

adf::Formulation& aafe::LinearFlanaganBelytschkoHexaFactory::BuildFormulation( 
  const aapc::SectionDefinition& sectionDefinition, 
  axis::domain::elements::ElementGeometry& geometry )
{
  real ahRatio = 0.1;
  if (sectionDefinition.IsPropertyDefined(_T("STABILIZATION_COEFFICIENT")))
  {
    aslse::ParameterValue& paramVal = sectionDefinition.GetPropertyValue(_T("STABILIZATION_COEFFICIENT"));
    ahRatio = aslse::ParameterSyntax::ToReal(paramVal);
  }
  return *new adf::LinearHexaFlanaganBelytschkoFormulation(ahRatio);
}
