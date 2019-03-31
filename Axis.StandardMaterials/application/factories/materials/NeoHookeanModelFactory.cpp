#include "NeoHookeanModelFactory.hpp"
#include "services/language/syntax/evaluation/AtomicValue.hpp"
#include "foundation/InvalidOperationException.hpp"
#include "services/language/syntax/evaluation/NumberValue.hpp"
#include "domain/materials/NeoHookeanModel.hpp"

namespace aafm  = axis::application::factories::materials;
namespace adm = axis::domain::materials;
namespace aslse = axis::services::language::syntax::evaluation;

// our name definitions
static const axis::String materialTypeName = _T("NEO_HOOKEAN");
static const axis::String poissonPropertyName = _T("POISSON");
static const axis::String elasticModulusPropertyName = _T("ELASTIC_MODULUS");
static const axis::String densityPropertyName = _T("LINEAR_DENSITY");

aafm::NeoHookeanModelFactory::NeoHookeanModelFactory( void )
{
	// nothing to do here
}

aafm::NeoHookeanModelFactory::~NeoHookeanModelFactory( void )
{
	// nothing to do here
}

bool aafm::NeoHookeanModelFactory::CanBuild( const axis::String& modelName, 
  const aslse::ParameterList& params ) const
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

	// check for the linear density definition
	if (!params.IsDeclared(densityPropertyName)) return false;
	if (!params.GetParameterValue(densityPropertyName).IsAtomic()) return false;
	if (!((aslse::AtomicValue&)params.GetParameterValue(
    densityPropertyName)).IsNumeric()) return false;

	// check if we have exactly only these parameters
	return params.Count() == 3;	
}

adm::MaterialModel& aafm::NeoHookeanModelFactory::Build( 
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
	double density = ((aslse::NumberValue&)params.GetParameterValue(
    densityPropertyName)).GetDouble();

	adm::NeoHookeanModel& neoHookeanModel = 
    *new adm::NeoHookeanModel(elasticModulus, poisson, density);
	return neoHookeanModel;
}

void aafm::NeoHookeanModelFactory::Destroy( void ) const
{
	delete this;
}
