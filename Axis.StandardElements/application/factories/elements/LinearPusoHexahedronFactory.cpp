#include "stdafx.h"
#include "LinearPusoHexahedronFactory.hpp"
#include "AxisString.hpp"
#include "application/parsing/core/SectionDefinition.hpp"
#include "domain/elements/ElementGeometry.hpp"
#include "domain/formulations/LinearHexahedralPusoFormulation.hpp"
#include "services/language/syntax/evaluation/ParameterSyntax.hpp"
#include "foundation/ArgumentException.hpp"

namespace aapc = axis::application::parsing::core;
namespace aafe = axis::application::factories::elements;
namespace ada = axis::domain::analyses;
namespace ade = axis::domain::elements;
namespace adi = axis::domain::integration;
namespace adf = axis::domain::formulations;
namespace aslse = axis::services::language::syntax::evaluation;

aafe::LinearPusoHexahedronFactory::LinearPusoHexahedronFactory( void )
{
  // nothing to do here
}

aafe::LinearPusoHexahedronFactory::~LinearPusoHexahedronFactory( void )
{
  // nothing to do here
}

bool aafe::LinearPusoHexahedronFactory::IsValidDefinition( const aapc::SectionDefinition& definition )
{
  bool canBuild = true;

  if (definition.GetSectionTypeName() != _T("LINEAR_HEXAHEDRON")) return false;

  if (!definition.IsPropertyDefined(_T("HOURGLASS_CONTROL"))) return false;
  aslse::ParameterValue& controlTypeParam = definition.GetPropertyValue(_T("HOURGLASS_CONTROL"));
  if (!aslse::ParameterSyntax::ToBoolean(controlTypeParam)) return false;

  if (!definition.IsPropertyDefined(_T("HOURGLASS_CONTROL_TYPE"))) return false;
  // check if the supplied value is valid
  axis::String value = definition.GetPropertyValue(_T("HOURGLASS_CONTROL_TYPE")).ToString();
  if (value != _T("ENHANCED")) return false;

  // check if only known properties and flags were supplied
  if (definition.PropertyCount() != 2 || definition.FlagCount() > 0) return false;

  // everything is ok
  return true;
}

adf::Formulation& aafe::LinearPusoHexahedronFactory::BuildFormulation( 
    const aapc::SectionDefinition&, axis::domain::elements::ElementGeometry& )
{
  return *new adf::LinearHexahedralPusoFormulation();
}
