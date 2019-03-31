// #include "MaterialFactoryTestFixture.hpp"
// #include "Services/Language/Syntax/Evaluation/ParameterList.hpp"
// #include "Services/Language/Syntax/Evaluation/NullValue.hpp"
// #include "Services/Language/Syntax/Evaluation/IdValue.hpp"
// #include "Services/Language/Syntax/Evaluation/NumberValue.hpp"
// #include "Application/Locators/StandardMaterialFactoryLocator.hpp"
// #include "Domain/Materials/Models/LinearElasticIsotropicModel.hpp"
// #include "foundation/BLAS/blas.hpp"
// 
// using namespace Axis::Services::Language::Syntax::Evaluation;
// using namespace Axis::Application::Locators;
// using namespace Axis::Domain::Materials::Models;
// using namespace Axis::foundation::BLAS;
// 
// void MaterialFactoryTestFixture::TestLinearIsoElasticCanBuildMethodFail( void )
// {
// 	LinearIsoElasticFactory factory;
// 	ParameterList& paramListFail = ParameterList::Create();
// 
// 	// should refuse with unknown material type
// 	CFIXCC_ASSERT_EQUALS(false, factory.CanBuild(_T("BOGUS_MATERIAL"), paramListFail));
// 
// 	// should refuse with empty list, even though material type is correct
// 	CFIXCC_ASSERT_EQUALS(false, factory.CanBuild(_T("LINEAR_ISO_ELASTIC"), paramListFail));
// 	
// 	// should refuse if missing params
// 	paramListFail.AddParameter(_T("POISSON"), *new NumberValue(0.3));
// 	CFIXCC_ASSERT_EQUALS(false, factory.CanBuild(_T("LINEAR_ISO_ELASTIC"), paramListFail));
// 
// 	// should refuse if we have more than enough params (excessive and unknown params)
// 	paramListFail.AddParameter(_T("ELASTIC_MODULUS"), *new NumberValue(200e6));
// 	paramListFail.AddParameter(_T("MY_DANGLING_PARAM"), *new NumberValue(0.0));
// 	CFIXCC_ASSERT_EQUALS(false, factory.CanBuild(_T("LINEAR_ISO_ELASTIC"), paramListFail));	
// }
// 
// void MaterialFactoryTestFixture::TestLinearIsoElasticCanBuildMethodPass( void )
// {
// 	LinearIsoElasticFactory factory;
// 	ParameterList& paramList = ParameterList::Create();
// 
// 	// should accept this without problems
// 	paramList.AddParameter(_T("POISSON"), *new NumberValue(0.3));
// 	paramList.AddParameter(_T("ELASTIC_MODULUS"), *new NumberValue(200e6));
// 	CFIXCC_ASSERT_EQUALS(true, factory.CanBuild(_T("LINEAR_ISO_ELASTIC"), paramList));
// }
// 
// void MaterialFactoryTestFixture::TestLinearIsoElasticBuildMethod( void )
// {
// 	const double nu = 0.3;
// 	const double E = 200e6;
// 
// 	LinearIsoElasticFactory factory;
// 	ParameterList& paramList = ParameterList::Create();
// 	paramList.AddParameter(_T("POISSON"), *new NumberValue(nu));
// 	paramList.AddParameter(_T("ELASTIC_MODULUS"), *new NumberValue(E));
// 
// 	FEMaterial& material = factory.Build(_T("LINEAR_ISO_ELASTIC"), paramList);
// 
// 	// must not be an empty elastic constitutive model, whereas the plastic model should be
// 	CFIXCC_ASSERT_EQUALS(false, material.ElasticModel().IsEmpty());
// 	CFIXCC_ASSERT_EQUALS(true, material.PlasticModel().IsEmpty());
// 	
// 	// check if constitutive model is not empty
// 	CFIXCC_ASSERT_EQUALS(false, material.ElasticModel().IsEmpty());
// 
// 	// check if material matrix is correct
// 	Matrix& matrix = material.ElasticModel().GetMaterialMatrix();
// 
// 	const double c11 = E*(1-nu) / ((1-2*nu)*(1+nu));
// 	const double c12 = E*nu / ((1-2*nu)*(1+nu));
// 	const double G   = E / (2*(1+nu));
// 
// 	// check if material matrix is well-formed
// 	CFIXCC_ASSERT_EQUALS((real)c11, matrix.GetElement(0,0));
// 	CFIXCC_ASSERT_EQUALS((real)c12, matrix.GetElement(0,1));
// 	CFIXCC_ASSERT_EQUALS((real)c12, matrix.GetElement(0,2));
// 	CFIXCC_ASSERT_EQUALS((real)c12, matrix.GetElement(1,0));
// 	CFIXCC_ASSERT_EQUALS((real)c11, matrix.GetElement(1,1));
// 	CFIXCC_ASSERT_EQUALS((real)c12, matrix.GetElement(1,2));
// 	CFIXCC_ASSERT_EQUALS((real)c12, matrix.GetElement(2,0));
// 	CFIXCC_ASSERT_EQUALS((real)c12, matrix.GetElement(2,1));
// 	CFIXCC_ASSERT_EQUALS((real)c11, matrix.GetElement(2,2));
// 	CFIXCC_ASSERT_EQUALS((real)G  , matrix.GetElement(3,3));
// 	CFIXCC_ASSERT_EQUALS((real)G  , matrix.GetElement(4,4));
// 	CFIXCC_ASSERT_EQUALS((real)G  , matrix.GetElement(5,5));
// 	CFIXCC_ASSERT_EQUALS((real)0  , matrix.GetElement(3,0));
// 	CFIXCC_ASSERT_EQUALS((real)0  , matrix.GetElement(3,1));
// 	CFIXCC_ASSERT_EQUALS((real)0  , matrix.GetElement(3,2));
// 	CFIXCC_ASSERT_EQUALS((real)0  , matrix.GetElement(4,0));
// 	CFIXCC_ASSERT_EQUALS((real)0  , matrix.GetElement(4,1));
// 	CFIXCC_ASSERT_EQUALS((real)0  , matrix.GetElement(4,2));
// 	CFIXCC_ASSERT_EQUALS((real)0  , matrix.GetElement(4,3));
// 	CFIXCC_ASSERT_EQUALS((real)0  , matrix.GetElement(5,0));
// 	CFIXCC_ASSERT_EQUALS((real)0  , matrix.GetElement(5,1));
// 	CFIXCC_ASSERT_EQUALS((real)0  , matrix.GetElement(5,2));
// 	CFIXCC_ASSERT_EQUALS((real)0  , matrix.GetElement(5,3));
// 	CFIXCC_ASSERT_EQUALS((real)0  , matrix.GetElement(5,4));
// 	CFIXCC_ASSERT_EQUALS((real)0  , matrix.GetElement(0,3));
// 	CFIXCC_ASSERT_EQUALS((real)0  , matrix.GetElement(1,3));
// 	CFIXCC_ASSERT_EQUALS((real)0  , matrix.GetElement(2,3));
// 	CFIXCC_ASSERT_EQUALS((real)0  , matrix.GetElement(0,4));
// 	CFIXCC_ASSERT_EQUALS((real)0  , matrix.GetElement(1,4));
// 	CFIXCC_ASSERT_EQUALS((real)0  , matrix.GetElement(2,4));
// 	CFIXCC_ASSERT_EQUALS((real)0  , matrix.GetElement(3,4));
// 	CFIXCC_ASSERT_EQUALS((real)0  , matrix.GetElement(0,5));
// 	CFIXCC_ASSERT_EQUALS((real)0  , matrix.GetElement(1,5));
// 	CFIXCC_ASSERT_EQUALS((real)0  , matrix.GetElement(2,5));
// 	CFIXCC_ASSERT_EQUALS((real)0  , matrix.GetElement(3,5));
// 	CFIXCC_ASSERT_EQUALS((real)0  , matrix.GetElement(4,5));
// 
// 	material.Destroy();
// }
// 
// void MaterialFactoryTestFixture::TestMaterialLocatorBuildMethod( void )
// {
// 	StandardMaterialFactoryLocator locator;
// 	
// 	// create and register factory
// 	LinearIsoElasticFactory *factory = new LinearIsoElasticFactory();
// 	locator.RegisterFactory(*factory);
// 
// 	// try to build through the material factory locator
// 	ParameterList& paramList = ParameterList::Create();
// 	paramList.AddParameter(_T("POISSON"), *new NumberValue(0.3));
// 	paramList.AddParameter(_T("ELASTIC_MODULUS"), *new NumberValue(200e6));
// 	FEMaterial& material = locator.BuildMaterial(_T("LINEAR_ISO_ELASTIC"), paramList);
// 
// 	// must not be an empty elastic constitutive model, whereas the plastic model should be
// 	CFIXCC_ASSERT_EQUALS(false, material.ElasticModel().IsEmpty());
// 	CFIXCC_ASSERT_EQUALS(true, material.PlasticModel().IsEmpty());
// 
// 	// check if constitutive model is not empty
// 	CFIXCC_ASSERT_EQUALS(false, material.ElasticModel().IsEmpty());
// 
// 	// unregister factory
// 	locator.UnregisterFactory(*factory);
// 	factory->Destroy();
// }