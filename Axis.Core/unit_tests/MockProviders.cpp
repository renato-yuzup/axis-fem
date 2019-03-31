#if defined DEBUG || defined _DEBUG

#include "unit_tests/unit_tests.hpp"

#include "MockProviders.hpp"
#include "application/parsing/parsers/PartParser.hpp"
#include "services/language/syntax/evaluation/ParameterList.hpp"
#include "services/language/syntax/evaluation/NullValue.hpp"
#include "services/language/syntax/evaluation/IdValue.hpp"
#include "services/language/syntax/evaluation/NumberValue.hpp"
#include "services/language/syntax/evaluation/ArrayValue.hpp"
#include "services/language/syntax/evaluation/ParameterAssignment.hpp"
#include "services/language/factories/IteratorFactory.hpp"
#include "services/language/iterators/InputIterator.hpp"
#include "AxisString.hpp"
#include "services/language/parsing/ParseResult.hpp"
#include "services/language/syntax/evaluation/AtomicValue.hpp"
#include "application/parsing/core/Sketchbook.hpp"
#include "domain/analyses/NumericalModel.hpp"

#include <assert.h>

using namespace axis::application::parsing::parsers;
using namespace axis::services::language::syntax::evaluation;
using namespace axis::services::language::factories;
using namespace axis::services::language::iterators;
using namespace axis::services::language::parsing;
using namespace axis::foundation;
using namespace axis::domain::analyses;
namespace aapc = axis::application::parsing::core;
namespace asmm = axis::services::messaging;
namespace asmg = axis::services::management;
namespace aapps = axis::application::parsing::parsers;
namespace aslse = axis::services::language::syntax::evaluation;
namespace adp = axis::domain::physics;
namespace ada = axis::domain::analyses;
namespace adm = axis::domain::materials;
namespace aafm = axis::application::factories::materials;
namespace afu = axis::foundation::uuids;
namespace afb = axis::foundation::blas;

/* ====================================================================================================================== */
/*                                                 AUXILIARY CLASSES                                                      */
/* ====================================================================================================================== */

void MockElementProvider::RegisterFactory( ElementParserFactory& factory )
{
	/* We are not going to use this method */
}

void MockElementProvider::UnregisterFactory( ElementParserFactory& factory )
{
	/* We are not going to use this method */
}

bool MockElementProvider::CanBuildElement( const aapc::SectionDefinition& sectionDefinition ) const
{
	if (sectionDefinition.GetSectionTypeName().compare(_T("TEST_ELEMENT")) == 0 &&
		sectionDefinition.IsPropertyDefined(_T("MY_PROP")))
	{
		_sucessfulCount++;
		return true;
	}
	_failedCount++;
	return false;
}

ElementParserFactory& MockElementProvider::GetFactory( 
  const aapc::SectionDefinition& sectionDefinition )
{
	throw std::exception("The method or operation is not implemented.");
}

ElementParserFactory& MockElementProvider::GetFactory( 
  const aapc::SectionDefinition& sectionDefinition ) const
{
	throw std::exception("The method or operation is not implemented.");
}
const char * MockElementProvider::GetFeaturePath( void ) const
{
	throw std::exception("The method or operation is not implemented.");
}

const char * MockElementProvider::GetFeatureName( void ) const
{
	throw std::exception("The method or operation is not implemented.");
}

void MockElementProvider::PostProcessRegistration( asmg::GlobalProviderCatalog& manager )
{
	/* We are not going to use this method */
}

void MockElementProvider::UnloadModule( asmg::GlobalProviderCatalog& manager )
{
	/* We are not going to use this method */
}

MockElementProvider::MockElementProvider( void )
{
	ResetCounters();
}

int MockElementProvider::GetFailedElementBuildQueryCount( void ) const
{
	return _failedCount;
}

int MockElementProvider::GetSuccessfulElementBuildQueryCount( void ) const
{
	return _sucessfulCount;
}

void MockElementProvider::ResetCounters( void )
{
	_sucessfulCount = 0;
	_failedCount = 0;
}

aapps::BlockParser& MockElementProvider::BuildVoidParser( void ) const
{
	throw std::exception("The method or operation is not implemented.");
}

aapps::BlockParser& MockElementProvider::BuildParser( const axis::String& contextName, 
                                                      const aslse::ParameterList& paramList )
{
	throw std::exception("The method or operation is not implemented.");
}

bool MockElementProvider::CanParse(const axis::String& blockName, 
                                   const aslse::ParameterList& paramList)
{
	throw std::exception("The method or operation is not implemented.");
}

void MockMaterialProvider::RegisterFactory( aafm::MaterialFactory& factory )
{
	/* We are not going to use this method */
}

void MockMaterialProvider::UnregisterFactory( aafm::MaterialFactory& factory )
{
	/* We are not going to use this method */
}

bool MockMaterialProvider::CanBuild( const axis::String& modelName, 
                                     const aslse::ParameterList& params ) const
{
	if (modelName.compare(_T("TEST_MAT"))!= 0)
	{
		_incorrectMaterialQueryCount++;
		return false;
	}
	if (params.Count() != 1 || !params.IsDeclared(_T("TEST_PROP")))
	{
		_invalidParamsQueryCount++;
		return false;
	}
	if (params.GetParameterValue(_T("TEST_PROP")).IsAtomic())
	{
		AtomicValue& val = (AtomicValue&)params.GetParameterValue(_T("TEST_PROP"));
		if (!val.IsNumeric()) return false;
		NumberValue& num = (NumberValue&)val;
		if (num.GetLong() != 1) return false;

		_successfulQueryCount++;
		return true;
	}
	return false;
}

adm::MaterialModel& MockMaterialProvider::BuildMaterial( const axis::String& modelName, 
                                                         const aslse::ParameterList& params )
{
	if (modelName.compare(_T("TEST_MAT"))!= 0)
	{
		Assert::Fail(_T("Unexpected material type."));
	}
	if (params.Count() != 1 || !params.IsDeclared(_T("TEST_PROP")))
	{
		Assert::Fail(_T("Unexpected material parameter."));
	}
	if (params.GetParameterValue(_T("TEST_PROP")).IsAtomic())
	{
		AtomicValue& val = (AtomicValue&)params.GetParameterValue(_T("TEST_PROP"));
		if (!val.IsNumeric())
			Assert::Fail(_T("Unexpected material parameter data type."));

		NumberValue& num = (NumberValue&)val;
		if (num.GetLong() != 1)
			Assert::Fail(_T("Unexpected material parameter value."));

		return *new TestMaterial();
	}
	Assert::Fail(_T("Unexpected material type or parameters."));
	throw "Code execution should never reach here.";
}

const char * MockMaterialProvider::GetFeaturePath( void ) const
{
	throw std::exception("The method or operation is not implemented.");
}

const char * MockMaterialProvider::GetFeatureName( void ) const
{
	throw std::exception("The method or operation is not implemented.");
}

void MockMaterialProvider::PostProcessRegistration( asmg::GlobalProviderCatalog& manager )
{
	/* We are not going to use this method */
}

void MockMaterialProvider::UnloadModule( asmg::GlobalProviderCatalog& manager )
{
	/* We are not going to use this method */
}

MockMaterialProvider::MockMaterialProvider( void )
{
	ResetCounters();
}

int MockMaterialProvider::GetIncorrectMaterialQueryCount( void ) const
{
	return _incorrectMaterialQueryCount;
}

int MockMaterialProvider::GetInvalidParamsQueryCount( void ) const
{
	return _invalidParamsQueryCount;
}

int MockMaterialProvider::GetSuccessfulQueryCount( void ) const
{
	return _successfulQueryCount;
}

void MockMaterialProvider::ResetCounters( void )
{
	_invalidParamsQueryCount = 0;
	_incorrectMaterialQueryCount = 0;
	_successfulQueryCount = 0;
}

bool MockProvider::CanParse( const axis::String& blockName, const aslse::ParameterList& paramList )
{
	throw std::exception("The method or operation is not implemented.");
}

aapps::BlockParser& MockProvider::BuildParser( const axis::String& contextName, 
                                               const aslse::ParameterList& paramList )
{
	throw std::exception("The method or operation is not implemented.");
}

const char * MockProvider::GetFeaturePath( void ) const
{
	throw std::exception("The method or operation is not implemented.");
}

const char * MockProvider::GetFeatureName( void ) const
{
	throw std::exception("The method or operation is not implemented.");
}

void MockProvider::PostProcessRegistration( asmg::GlobalProviderCatalog& manager )
{
	/* We are not going to use this method */
}

void MockProvider::UnloadModule( asmg::GlobalProviderCatalog& manager )
{
	/* We are not going to use this method */
}

void MockContext::AddUndefinedCustomSymbol( const axis::String& name, const axis::String& type )
{
	_crossRefCount++;
}

void MockContext::DefineCustomSymbol( const axis::String& name, const axis::String& type )
{
	// do nothing
}

ParseContext::RunMode MockContext::GetRunMode( void ) const
{
	return ParseContext::kTrialMode;
}

void MockContext::RegisterEvent( asmm::EventMessage& event )
{
	++_eventsCount;
}

MockContext::MockContext( void )
{
	_eventsCount = 0;
	_crossRefCount = 0;
}

int MockContext::GetRegisteredEventsCount( void ) const
{
	return _eventsCount;
}

void MockContext::ClearRegisteredEvents( void )
{
	_eventsCount = 0;
}

int MockContext::GetCrossRefCount( void ) const
{
	return _crossRefCount;
}

void MockContext::ClearCrossRefs( void )
{
	_crossRefCount = 0;
}

TestMaterial::TestMaterial( void ) : MaterialModel((real)1.0, 1)
{
  // nothing to do here
}

bool TestMaterial::IsTestMaterial( void ) const
{
	return true;
}

void TestMaterial::Destroy( void ) const
{
	delete this;
}

afb::DenseMatrix& TestMaterial::GetMaterialTensor( void ) const
{
	throw std::exception("The method or operation is not implemented.");
}

MaterialModel& TestMaterial::Clone( int ) const
{
	return *new TestMaterial();
}

void TestMaterial::UpdateStresses( adp::UpdatedPhysicalState&,
  const adp::InfinitesimalState&, const ada::AnalysisTimeline&, int )
{
	// nothing to do here
}

real TestMaterial::GetWavePropagationSpeed( void ) const
{
	return 1;
}

real TestMaterial::GetBulkModulus( void ) const
{
  return 1;
}

real TestMaterial::GetShearModulus( void ) const
{
  return 1;
}

afu::Uuid TestMaterial::GetTypeId( void ) const
{
  throw std::exception("The method or operation is not implemented.");
}

#endif
