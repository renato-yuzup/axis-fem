// #include "LinearHexahedronTestFixture.hpp"
// #include "Application/Factories/ShapeFunctions/LinearShapeFunctionFactory.hpp"
// #include "Domain/ShapeFunctions/ShapeFunction.hpp"
// #include "Domain/Materials/Models/LinearElasticIsotropicModel.hpp"
// #include "Domain/Materials/Base/FEMaterial.hpp"
// #include "foundation/BLAS/blas.hpp"
// #include "Domain/Formulations/IsoparametricFormulation.hpp"
// #include "Domain/Materials/Base/FEMaterial.hpp"
// #include "Domain/Materials/Models/LinearElasticIsotropicModel.hpp"
// #include "Domain/Elements/LinearHexahedralShape.hpp"
// #include "Application/Factories/Integration/LinearElementGaussPointFactory.hpp"
// 
// using namespace Axis::Application::Factories::ShapeFunctions;
// using namespace Axis::Application::Factories::Integration;
// using namespace Axis::Domain::ShapeFunctions;
// using namespace Axis::Domain::Formulations;
// using namespace Axis::Domain::Materials::Base;
// using namespace Axis::Domain::Materials::Models;
// using namespace Axis::foundation::Math;
// using namespace Axis::foundation::BLAS;
// 
// #ifdef AXIS_DOUBLEPRECISION
// #define ZERO_TOLERANCE			1e-9
// #define D(XCoordinate)					XCoordinate
// #define ASSERT_F(XCoordinate,YCoordinate)			CFIXCC_ASSERT_LESS_OR_EQUAL((double)abs((XCoordinate)-(YCoordinate)), ZERO_TOLERANCE)
// #else
// #define ZERO_TOLERANCE			1e-6f
// #define D(XCoordinate)					XCoordinate##f
// #define ASSERT_F(XCoordinate,YCoordinate)			CFIXCC_ASSERT_LESS_OR_EQUAL((float)abs((XCoordinate)-(YCoordinate)), ZERO_TOLERANCE)
// #endif
// 
// void LinearHexahedronTestFixture::TestShapeFunctionCoefficients( void )
// {
// 	LinearShapeFunctionFactory& factory = LinearShapeFunctionFactory::GetInstance();
// 
// 	// first, check that the factory is always returning the same
// 	// instance
// 	ShapeFunction *s1 = &factory.BuildHexahedronShapeFunction(0);
// 	ShapeFunction *s2 = &factory.BuildHexahedronShapeFunction(0);
// 	CFIXCC_ASSERT_EQUALS(s1, s2);
// 
// 	// now, check correctness of each instance
// 
// 	real vals1[3] = {0.5, 0.0, 0.0};
// 	real vals2[3] = {0.0, 2.0, 0.0};
// 	real vals3[3] = {0.0, 0.0, -1.0};
// 
// 	ShapeFunction *s = &factory.BuildHexahedronShapeFunction(0);
// 	ASSERT_F(1 / 16.0, s->SolveNumeric(vals1));
// 	ASSERT_F(-1 / 8.0, s->SolveNumeric(vals2));
// 	ASSERT_F(1 / 4.0, s->SolveNumeric(vals3));
// 
// 	s = &factory.BuildHexahedronShapeFunction(1);
// 	ASSERT_F(1 / 8.0 * 1.5, s->SolveNumeric(vals1));
// 	ASSERT_F(-1 / 8.0, s->SolveNumeric(vals2));
// 	ASSERT_F(1 / 4.0, s->SolveNumeric(vals3));
// 
// 	s = &factory.BuildHexahedronShapeFunction(2);
// 	ASSERT_F(1/8.0 * 1.5, s->SolveNumeric(vals1));
// 	ASSERT_F(1/8.0 * 3.0, s->SolveNumeric(vals2));
// 	ASSERT_F(1 / 4.0, s->SolveNumeric(vals3));
// 
// 	s = &factory.BuildHexahedronShapeFunction(3);
// 	ASSERT_F(1 / 16.0, s->SolveNumeric(vals1));
// 	ASSERT_F(1/8.0 * 3.0, s->SolveNumeric(vals2));
// 	ASSERT_F(1 / 4.0, s->SolveNumeric(vals3));
// 
// 	s = &factory.BuildHexahedronShapeFunction(4);
// 	ASSERT_F(1 / 16.0, s->SolveNumeric(vals1));
// 	ASSERT_F(-1 / 8.0, s->SolveNumeric(vals2));
// 	ASSERT_F(0.0, s->SolveNumeric(vals3));
// 
// 	s = &factory.BuildHexahedronShapeFunction(5);
// 	ASSERT_F(1/8.0 * 1.5, s->SolveNumeric(vals1));
// 	ASSERT_F(-1 / 8.0, s->SolveNumeric(vals2));
// 	ASSERT_F(0.0, s->SolveNumeric(vals3));
// 
// 	s = &factory.BuildHexahedronShapeFunction(6);
// 	ASSERT_F(1/8.0 * 1.5, s->SolveNumeric(vals1));
// 	ASSERT_F(1/8.0 * 3.0, s->SolveNumeric(vals2));
// 	ASSERT_F(0.0, s->SolveNumeric(vals3));
// 
// 	s = &factory.BuildHexahedronShapeFunction(7);
// 	ASSERT_F(1 / 16.0, s->SolveNumeric(vals1));
// 	ASSERT_F(1/8.0 * 3.0, s->SolveNumeric(vals2));
// 	ASSERT_F(0.0, s->SolveNumeric(vals3));
// }
// 
// void LinearHexahedronTestFixture::TestShapeFunctionDifferential( void )
// {
// 	LinearShapeFunctionFactory& factory = LinearShapeFunctionFactory::GetInstance();
// 
// 	real vals1[3] = {0.9, 0.5, -0.5};
// 	real vals2[3] = {-0.5, 0.9, 0.5};
// 	real vals3[3] = {0.5, -0.5, 0.9};
// 
// 	ShapeFunction *s = &factory.BuildHexahedronShapeFunction(0);
// 	ASSERT_F(-1 / 16.0 * 1.5, s->SolveDifferentialNumeric(XCoordinate, vals1));
// 	ASSERT_F(-1/8.0*1.5*0.5, s->SolveDifferentialNumeric(YCoordinate, vals2));
// 	ASSERT_F(-1/8.0*0.5*1.5, s->SolveDifferentialNumeric(ZCoordinate, vals3));
// 
// 	s = &factory.BuildHexahedronShapeFunction(1);
// 	ASSERT_F(1 / 8.0*0.5*1.5, s->SolveDifferentialNumeric(XCoordinate, vals1));
// 	ASSERT_F(-1/8.0*0.5*0.5, s->SolveDifferentialNumeric(YCoordinate, vals2));
// 	ASSERT_F(-1/8.0*1.5*1.5, s->SolveDifferentialNumeric(ZCoordinate, vals3));
// 
// 	s = &factory.BuildHexahedronShapeFunction(2);
// 	ASSERT_F(1/8.0*1.5*1.5, s->SolveDifferentialNumeric(XCoordinate, vals1));
// 	ASSERT_F(1/8.0*0.5*0.5, s->SolveDifferentialNumeric(YCoordinate, vals2));
// 	ASSERT_F(-1/8.0*1.5*0.5, s->SolveDifferentialNumeric(ZCoordinate, vals3));
// 
// 	s = &factory.BuildHexahedronShapeFunction(3);
// 	ASSERT_F(-1/8.0*1.5*1.5, s->SolveDifferentialNumeric(XCoordinate, vals1));
// 	ASSERT_F(1/8.0*1.5*0.5, s->SolveDifferentialNumeric(YCoordinate, vals2));
// 	ASSERT_F(-1/8.0*0.5*0.5, s->SolveDifferentialNumeric(ZCoordinate, vals3));
// 
// 	s = &factory.BuildHexahedronShapeFunction(4);
// 	ASSERT_F(-1/8.0*0.5*0.5, s->SolveDifferentialNumeric(XCoordinate, vals1));
// 	ASSERT_F(-1/8.0*1.5*1.5, s->SolveDifferentialNumeric(YCoordinate, vals2));
// 	ASSERT_F(1/8.0*0.5*1.5, s->SolveDifferentialNumeric(ZCoordinate, vals3));
// 
// 	s = &factory.BuildHexahedronShapeFunction(5);
// 	ASSERT_F(1/8.0*0.5*0.5, s->SolveDifferentialNumeric(XCoordinate, vals1));
// 	ASSERT_F(-1/8.0*0.5*1.5, s->SolveDifferentialNumeric(YCoordinate, vals2));
// 	ASSERT_F(1/8.0*1.5*1.5, s->SolveDifferentialNumeric(ZCoordinate, vals3));
// 
// 	s = &factory.BuildHexahedronShapeFunction(6);
// 	ASSERT_F(1/8.0*1.5*0.5, s->SolveDifferentialNumeric(XCoordinate, vals1));
// 	ASSERT_F(1/8.0*0.5*1.5, s->SolveDifferentialNumeric(YCoordinate, vals2));
// 	ASSERT_F(1/8.0*1.5*0.5, s->SolveDifferentialNumeric(ZCoordinate, vals3));
// 
// 	s = &factory.BuildHexahedronShapeFunction(7);
// 	ASSERT_F(-1/8.0*1.5*0.5, s->SolveDifferentialNumeric(XCoordinate, vals1));
// 	ASSERT_F(1/8.0*1.5*1.5, s->SolveDifferentialNumeric(YCoordinate, vals2));
// 	ASSERT_F(1/8.0*0.5*0.5, s->SolveDifferentialNumeric(ZCoordinate, vals3));
// }
// 
// void LinearHexahedronTestFixture::TestLinearIsoElasticMaterial( void )
// {
// 	const real E = 200e9;
// 	const real nu = 0.3;
// 	Axis::Domain::Materials::Models::LinearElasticIsotropicModel& elasticModel = *new Axis::Domain::Materials::Models::LinearElasticIsotropicModel(E, nu);
// 	Axis::Domain::Materials::Base::FEMaterial material(elasticModel);
// 
// 	Matrix& matrix = material.ElasticModel().GetMaterialMatrix();
// 
// 	// assert matrix general characteristics
// 	CFIXCC_ASSERT_EQUALS(6, (int)matrix.Rows());
// 	CFIXCC_ASSERT_EQUALS(6, (int)matrix.Columns());
// 
// 	// check elements
// 	real c11 = E*(1-nu) / ((1-2*nu)*(1+nu));
// 	real c12 = E*nu / ((1-2*nu)*(1+nu));
// 	real G  = E / (2*(1+nu));
// 
// 	ASSERT_F(c11, matrix.GetElement(0, 0));
// 	ASSERT_F(c12, matrix.GetElement(0, 1));
// 	ASSERT_F(c12, matrix.GetElement(0, 2));
// 	ASSERT_F(0, matrix.GetElement(0, 3));
// 	ASSERT_F(0, matrix.GetElement(0, 4));
// 	ASSERT_F(0, matrix.GetElement(0, 5));
// 
// 	ASSERT_F(c12, matrix.GetElement(1, 0));
// 	ASSERT_F(c11, matrix.GetElement(1, 1));
// 	ASSERT_F(c12, matrix.GetElement(1, 2));
// 	ASSERT_F(0, matrix.GetElement(1, 3));
// 	ASSERT_F(0, matrix.GetElement(1, 4));
// 	ASSERT_F(0, matrix.GetElement(1, 5));
// 
// 	ASSERT_F(c12, matrix.GetElement(2, 0));
// 	ASSERT_F(c12, matrix.GetElement(2, 1));
// 	ASSERT_F(c11, matrix.GetElement(2, 2));
// 	ASSERT_F(0, matrix.GetElement(2, 3));
// 	ASSERT_F(0, matrix.GetElement(2, 4));
// 	ASSERT_F(0, matrix.GetElement(2, 5));
// 
// 	ASSERT_F(0, matrix.GetElement(3, 0));
// 	ASSERT_F(0, matrix.GetElement(3, 1));
// 	ASSERT_F(0, matrix.GetElement(3, 2));
// 	ASSERT_F(G, matrix.GetElement(3, 3));
// 	ASSERT_F(0, matrix.GetElement(3, 4));
// 	ASSERT_F(0, matrix.GetElement(3, 5));
// 
// 	ASSERT_F(0, matrix.GetElement(4, 0));
// 	ASSERT_F(0, matrix.GetElement(4, 1));
// 	ASSERT_F(0, matrix.GetElement(4, 2));
// 	ASSERT_F(0, matrix.GetElement(4, 3));
// 	ASSERT_F(G, matrix.GetElement(4, 4));
// 	ASSERT_F(0, matrix.GetElement(4, 5));
// 
// 	ASSERT_F(0, matrix.GetElement(5, 0));
// 	ASSERT_F(0, matrix.GetElement(5, 1));
// 	ASSERT_F(0, matrix.GetElement(5, 2));
// 	ASSERT_F(0, matrix.GetElement(5, 3));
// 	ASSERT_F(0, matrix.GetElement(5, 4));
// 	ASSERT_F(G, matrix.GetElement(5, 5));
// }
// 
// void LinearHexahedronTestFixture::TestStiffnessMatrix( void )
// {
// }
// 
// void LinearHexahedronTestFixture::TestMassMatrix( void )
// {
// }
// 
// FiniteElement& LinearHexahedronTestFixture::CreateTestElement( void )
// {
// 	GaussLegendreQuadrature3D& numIntegr = CreateNumericalIntegration();
// 	return *new FiniteElement(1, 1, LinearHexahedralShape::GetInstance(), numIntegr.ElementGeometry(), numIntegr.GetFormulation());
// }
// 
// GaussLegendreQuadrature3D& LinearHexahedronTestFixture::CreateNumericalIntegration( void )
// {
// 	real E = 200E9;
// 	real nu = 0.3;
// 	Geometry& g = *new Geometry(8, 6, 8, *new IntegrationDimension(27));
// 
// 	LinearElementGaussPointFactory factory;
// 	IntegrationDimension &set = factory.CreateHexahedronVolumeIntegrationPoints(LinearElementGaussPointFactory::Full);
// 
// 	FEMaterial& material = *new FEMaterial(*new LinearElasticIsotropicModel(E, nu));
// 	GaussLegendreQuadrature3D& numIntegr = *new GaussLegendreQuadrature3D(g);
// 	IsoparametricFormulation& formulation = *new IsoparametricFormulation(material, numIntegr);
// 	numIntegr.SetFormulation(formulation);
// 	return numIntegr;
// }
// 
// void LinearHexahedronTestFixture::TestIsoparametricJacobian( void )
// {
// }
// 
// void LinearHexahedronTestFixture::TestIsoparametricInverseJacobian( void )
// {
// 	DenseMatrix J(3, 3);
// 	DenseMatrix Jinv(3,3);
// 
// 	J.SetElement(0, 0, 1);
// 	J.SetElement(0, 1, 6);
// 	J.SetElement(0, 2, 1);
// 	J.SetElement(1, 0, 3);
// 	J.SetElement(1, 1, 5);
// 	J.SetElement(1, 2, 1);
// 	J.SetElement(2, 0, 2);
// 	J.SetElement(2, 1, 7);
// 	J.SetElement(2, 2, 4);
// 
// 	GaussLegendreQuadrature3D& numIntegr = CreateNumericalIntegration();
// 
// 	numIntegr.CalculateJacobianInverse(Jinv, J);
// 	ASSERT_F(-0.361111111111111, Jinv.GetElement(0,0));
// 	ASSERT_F( 0.472222222222222, Jinv.GetElement(0,1));
// 	ASSERT_F(-0.027777777777778, Jinv.GetElement(0,2));
// 	ASSERT_F( 0.277777777777778, Jinv.GetElement(1,0));
// 	ASSERT_F(-0.055555555555556, Jinv.GetElement(1,1));
// 	ASSERT_F(-0.055555555555556, Jinv.GetElement(1,2));
// 	ASSERT_F(-0.305555555555556, Jinv.GetElement(2,0));
// 	ASSERT_F(-0.138888888888889, Jinv.GetElement(2,1));
// 	ASSERT_F( 0.361111111111111, Jinv.GetElement(2,2));
// }
// 
// void LinearHexahedronTestFixture::TestIsoparametricJacobianDeterminant( void )
// {
// 	DenseMatrix J(3, 3);
// 
// 	J.SetElement(0, 0, 1);
// 	J.SetElement(0, 1, 6);
// 	J.SetElement(0, 2, 1);
// 	J.SetElement(1, 0, 3);
// 	J.SetElement(1, 1, 5);
// 	J.SetElement(1, 2, 1);
// 	J.SetElement(2, 0, 2);
// 	J.SetElement(2, 1, 7);
// 	J.SetElement(2, 2, 4);
// 
// 	GaussLegendreQuadrature3D& numIntegr = CreateNumericalIntegration();
// 
// 	real detJ = numIntegr.CalculateJacobianDeterminant(J);
// 	ASSERT_F(-36, detJ);
// }