#include "BiLinearPlasticityModelFactory.hpp"
#include "services/language/syntax/evaluation/AtomicValue.hpp"
#include "foundation/InvalidOperationException.hpp"
#include "services/language/syntax/evaluation/NumberValue.hpp"
#include "domain/materials/BiLinearPlasticityModel.hpp"

namespace aafm  = axis::application::factories::materials;
namespace adm = axis::domain::materials;
namespace aslse = axis::services::language::syntax::evaluation;

// our name definitions
static const axis::String materialTypeName = _T("BILINEAR_PLASTICITY");
static const axis::String poissonPropertyName = _T("POISSON");
static const axis::String elasticModulusPropertyName = _T("ELASTIC_MODULUS");
static const axis::String yieldStressName = _T("YIELD_STRESS");
static const axis::String hardeningCoefficientName = _T("HARDENING_COEFFICIENT");
static const axis::String densityPropertyName = _T("LINEAR_DENSITY");

aafm::BiLinearPlasticityModelFactory::BiLinearPlasticityModelFactory( void )
{
  // nothing to do here
}

aafm::BiLinearPlasticityModelFactory::~BiLinearPlasticityModelFactory( void )
{
  // nothing to do here
}

bool aafm::BiLinearPlasticityModelFactory::CanBuild( 
  const axis::String& modelName, const aslse::ParameterList& params ) const
{
  // check if we have the correct params
  if (modelName != materialTypeName) return false;

  // check for the Poisson ratio definition
  if (!params.IsDeclared(poissonPropertyName)) return false;
  if (!params.GetParameterValue(poissonPropertyName).IsAtomic()) return false;
  if (!((aslse::AtomicValue&)params.GetParameterValue(
    poissonPropertyName)).IsNumeric()) return false;

  // check for the elastic modulus definition
  if (!params.IsDeclared(elasticModulusPropertyName)) return false;
  if (!params.GetParameterValue(elasticModulusPropertyName).IsAtomic()) 
    return false;
  if (!((aslse::AtomicValue&)params.GetParameterValue(
    elasticModulusPropertyName)).IsNumeric()) return false;

  // check for the hardening coefficient definition
  if (!params.IsDeclared(hardeningCoefficientName)) return false;
  if (!params.GetParameterValue(hardeningCoefficientName).IsAtomic()) 
    return false;
  if (!((aslse::AtomicValue&)params.GetParameterValue(
    hardeningCoefficientName)).IsNumeric()) return false;

  // check for the yield stress definition
  if (!params.IsDeclared(yieldStressName)) return false;
  if (!params.GetParameterValue(yieldStressName).IsAtomic()) 
    return false;
  if (!((aslse::AtomicValue&)params.GetParameterValue(
    yieldStressName)).IsNumeric()) return false;

  // check for the linear density definition
  if (!params.IsDeclared(densityPropertyName)) return false;
  if (!params.GetParameterValue(densityPropertyName).IsAtomic()) return false;
  if (!((aslse::AtomicValue&)params.GetParameterValue(
    densityPropertyName)).IsNumeric()) return false;

  // check if we have exactly only these parameters
  return params.Count() == 5;	
}

adm::MaterialModel& aafm::BiLinearPlasticityModelFactory::Build( 
  const axis::String& modelName, const aslse::ParameterList& params )
{
  if (!CanBuild(modelName, params))
  {	
    throw axis::foundation::InvalidOperationException();
  }

  // get property values
  double poisson = ((aslse::NumberValue&)params.GetParameterValue(
    poissonPropertyName)).GetDouble();
  double elasticModulus = ((aslse::NumberValue&)params.GetParameterValue(
    elasticModulusPropertyName)).GetDouble();
  double yieldStress = ((aslse::NumberValue&)params.GetParameterValue(
    yieldStressName)).GetDouble();
  double hCoeff = ((aslse::NumberValue&)params.GetParameterValue(
    hardeningCoefficientName)).GetDouble();
  double density = ((aslse::NumberValue&)params.GetParameterValue(
    densityPropertyName)).GetDouble();

  adm::BiLinearPlasticityModel& plasticityModel = 
    *new adm::BiLinearPlasticityModel(elasticModulus, poisson, yieldStress, 
    hCoeff, density, 1);
  return plasticityModel;
}

void aafm::BiLinearPlasticityModelFactory::Destroy( void ) const
{
  delete this;
}
