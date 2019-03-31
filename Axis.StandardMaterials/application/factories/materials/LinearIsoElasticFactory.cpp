#include "LinearIsoElasticFactory.hpp"
#include "services/language/syntax/evaluation/AtomicValue.hpp"
#include "foundation/InvalidOperationException.hpp"
#include "services/language/syntax/evaluation/NumberValue.hpp"
#include "domain/materials/LinearElasticIsotropicModel.hpp"

using namespace axis::services::language::syntax::evaluation;
// our name definitions
static const axis::String materialTypeName = _T("LINEAR_ISO_ELASTIC");
static const axis::String poissonPropertyName = _T("POISSON");
static const axis::String elasticModulusPropertyName = _T("ELASTIC_MODULUS");
static const axis::String densityPropertyName = _T("LINEAR_DENSITY");

axis::application::factories::materials::LinearIsoElasticFactory::LinearIsoElasticFactory( void )
{
	// nothing to do here
}

axis::application::factories::materials::LinearIsoElasticFactory::~LinearIsoElasticFactory( void )
{
	// nothing to do here
}

bool axis::application::factories::materials::LinearIsoElasticFactory::CanBuild( const axis::String& modelName, const axis::services::language::syntax::evaluation::ParameterList& params ) const
{
	// check if we have the correct params
	if (modelName != materialTypeName) return false;
	
	// check for the Poisson ratio definition
	if (!params.IsDeclared(poissonPropertyName)) return false;
	if (!params.GetParameterValue(poissonPropertyName).IsAtomic()) return false;
	if (!((AtomicValue&)params.GetParameterValue(poissonPropertyName)).IsNumeric()) return false;
	
	
	// check for the elastic modulus definition
	if (!params.IsDeclared(elasticModulusPropertyName)) return false;
	if (!params.GetParameterValue(elasticModulusPropertyName).IsAtomic()) return false;
	if (!((AtomicValue&)params.GetParameterValue(elasticModulusPropertyName)).IsNumeric()) return false;

	// check for the linear density definition
	if (!params.IsDeclared(densityPropertyName)) return false;
	if (!params.GetParameterValue(densityPropertyName).IsAtomic()) return false;
	if (!((AtomicValue&)params.GetParameterValue(densityPropertyName)).IsNumeric()) return false;

	// check if we have exactly only these parameters
	return params.Count() == 3;	
}

axis::domain::materials::MaterialModel& axis::application::factories::materials::LinearIsoElasticFactory::Build( const axis::String& modelName, const axis::services::language::syntax::evaluation::ParameterList& params )
{
	if (!CanBuild(modelName, params))
	{	
		throw axis::foundation::InvalidOperationException();
	}

	// get property values
	double poisson = ((NumberValue&)params.GetParameterValue(poissonPropertyName)).GetDouble();
	double elasticModulus = ((NumberValue&)params.GetParameterValue(elasticModulusPropertyName)).GetDouble();
	double density = ((NumberValue&)params.GetParameterValue(densityPropertyName)).GetDouble();

	axis::domain::materials::LinearElasticIsotropicModel& elasticModel = *new axis::domain::materials::LinearElasticIsotropicModel(elasticModulus, poisson, density);
	return elasticModel;
}

void axis::application::factories::materials::LinearIsoElasticFactory::Destroy( void ) const
{
	delete this;
}
