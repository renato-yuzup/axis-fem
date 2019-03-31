// #if defined _DEBUG || defined DEBUG
// 
// #include "unit_tests.hpp"
// #include "application/factories/elements/LinearSimpleHexahedronFactory.hpp"
// #include "application/factories/integration/LinearElementGaussPointFactory.hpp"
// #include "application/factories/shape_functions/LinearShapeFunctionFactory.hpp"
// #include "domain/analyses/AnalysisTimeline.hpp"
// #include "domain/analyses/NumericalModel.hpp"
// #include "domain/elements/MatrixOption.hpp"
// #include "domain/elements/FiniteElement.hpp"
// #include "domain/integration/GaussLegendreQuadrature3D.hpp"
// #include "domain/formulations/IsoparametricFormulation.hpp"
// #include "domain/elements/LinearHexahedralShape.hpp"
// #include "domain/physics/InfinitesimalState.hpp"
// #include "MockMaterial.hpp"
// #include "foundation/blas/AnyVector.hpp"
// 
// using namespace axis::foundation::blas;
// using namespace axis::domain::analyses;
// using namespace axis::domain::elements;
// using namespace axis::domain::formulations;
// using namespace axis::domain::integration;
// using namespace axis::domain::materials;
// using namespace axis::domain::physics;
// using namespace axis::application::factories::elements;
// using namespace axis::application::factories::integration;
// using namespace axis::application::factories::shape_functions;
// 
// namespace axis { namespace unit_tests { namespace AxisStandardElements {
// 
// 	TEST_CLASS(LinearGaussLegendreQuadrature3DTest)
// 	{
// 	private:
// 		NumericalModel& BuildSimpleHexahedralElement(void) const
// 		{
// 			LinearSimpleHexahedronFactory factory;
// 			LinearElementGaussPointFactory gpf;
//       LinearShapeFunctionFactory& lsff = LinearShapeFunctionFactory::GetInstance();
// 
// 			NumericalModel& model = NumericalModel::Create();
// 
// 			// create integration points
// 			IntegrationDimension& d = gpf.CreateHexahedronVolumeIntegrationPoints(LinearElementGaussPointFactory::Full);
// 
// 			ElementGeometry& g = *new ElementGeometry(8, 6, 12, d);
// 			
// 			// create nodes
// 			Node *nodes[8];
// 			nodes[0] = new Node(0, 0, 0, 0, 0);
// 			nodes[1] = new Node(1, 1, 1, 0, 0);
// 			nodes[2] = new Node(2, 2, 1, 1, 0);
// 			nodes[3] = new Node(3, 3, 0, 1, 0);
// 			nodes[4] = new Node(4, 4, 0, 0, 1);
// 			nodes[5] = new Node(5, 5, 1, 0, 1);
// 			nodes[6] = new Node(6, 6, 1, 1, 1);
// 			nodes[7] = new Node(7, 7, 0, 1, 1);
// 			for (int i = 0; i < 8; i++)
// 			{
// 				model.Nodes().Add(*nodes[i]);
//         nodes[i]->InitDofs(3, i*3);
//         g.SetNode(i, *nodes[i]);
//         g.SetShapeFunction(i, lsff.BuildHexahedronShapeFunction(i));
// 			}
// 
// 			ElementNumericalIntegration& nim = *new GaussLegendreQuadrature3D(g);
// 			MaterialModel& material = *new MockMaterial();
// 			material.Density() = 7850;
// 			Formulation& f = *new IsoparametricFormulation(nim);
// 
// 			FiniteElement& fe = *new FiniteElement(0, g, material, f);
// 			model.Elements().Add(fe);
//       model.Kinematics().ResetAll(24);
// 			return model;
// 		}
// 
// 	public:
// 		TEST_METHOD(TestCalculateFaceArea)
// 		{
// 		}
// 		TEST_METHOD(GetCharacteristicLengthTest)
// 		{
// 			ColumnVector v(8);
// 			NumericalModel& model = BuildSimpleHexahedralElement();
// 			FiniteElement& fe = model.Elements().GetByInternalIndex(0);
// 			real length = fe.GetCriticalTimestep(v);
// 
// 			Assert::AreEqual((real)1.0, length, REAL_ROUNDOFF_TOLERANCE);
// 		}
// 		TEST_METHOD(TestCalculateStiffnessMatrix)
// 		{
// 
// 		}
// 		TEST_METHOD(TestCalculateConsistentMassMatrix)
// 		{
// 
// 		}
// 		TEST_METHOD(CalculateLumpedMassMatrixTest)
// 		{
// 			NumericalModel& model = BuildSimpleHexahedralElement();
// 			FiniteElement& fe = model.Elements().GetByInternalIndex(0);
// 			
// 			fe.UpdateMatrices(LumpedMassOnlyOption(), model.Kinematics().Displacement(), 
//                         model.Kinematics().Displacement());
// 			const Vector& lumped = fe.GetLumpedMass();
// 
// 			// for a unity cube, the lumped matrix has all entries the same
// 			for (int i = 0; i < 24; i++)
// 			{
// 				real val = lumped(i);
// 				Assert::AreEqual((real)981.2499999999999, val, REAL_ROUNDOFF_TOLERANCE);
// 			}
// 		}
// 		TEST_METHOD(CalculateInternalForcesTest)
// 		{
// 			NumericalModel& model = BuildSimpleHexahedralElement();
// 			FiniteElement& fe = model.Elements().GetByInternalIndex(0);
// 			IntegrationDimension& points = fe.Geometry().GetIntegrationPoints();
//       AnalysisTimeline& ti = AnalysisTimeline::Create(0,0,0,0);
// 
// 			// update stresses in integration points
// 			fe.InitializeForAnalysis();
// 			for (int i = 0; i < points.Count(); i++)
// 			{
// 				InfinitesimalState& d = points[i].State();
//         d.Reset();
// 				Vector& stress = d.Stress();
// 
// 				stress.ClearAll();
// 				stress(0) = 1e5;
// 				stress(1) = -1e5;
// 				stress(2) = 5e5;
// 				stress(3) = 2.5e4;
// 				stress(4) = -1.25e4;
// 				stress(5) = 0;
// 			}
// 
// 			ColumnVector fint(24);
// 			ColumnVector bogusVector(24);
// 			fe.ExtractInternalForce(fint, bogusVector, bogusVector, bogusVector, ti);
// 
// 			// check results
// 			Assert::AreEqual((real) 3.125000000000000e+04, fint(0),  REAL_TOLERANCE*1e4);
// 			Assert::AreEqual((real)-2.187500000000000e+04, fint(1),  REAL_TOLERANCE*1e4);
// 			Assert::AreEqual((real) 1.218750000000000e+05, fint(2),  REAL_TOLERANCE*1e5);
// 			Assert::AreEqual((real)-1.875000000000000e+04, fint(3),  REAL_TOLERANCE*1e4);
// 			Assert::AreEqual((real)-3.437500000000000e+04, fint(4),  REAL_TOLERANCE*1e4);
// 			Assert::AreEqual((real) 1.218750000000000e+05, fint(5),  REAL_TOLERANCE*1e5);
// 			Assert::AreEqual((real)-3.125000000000000e+04, fint(6),  REAL_TOLERANCE*1e4);
// 			Assert::AreEqual((real) 1.562500000000000e+04, fint(7),  REAL_TOLERANCE*1e4);
// 			Assert::AreEqual((real) 1.281250000000000e+05, fint(8),  REAL_TOLERANCE*1e5);
// 			Assert::AreEqual((real) 1.875000000000000e+04, fint(9),  REAL_TOLERANCE*1e4);
// 			Assert::AreEqual((real) 2.812500000000000e+04, fint(10), REAL_TOLERANCE*1e4);
// 			Assert::AreEqual((real) 1.281250000000000e+05, fint(11), REAL_TOLERANCE*1e5);
// 			Assert::AreEqual((real) 3.125000000000000e+04, fint(12), REAL_TOLERANCE*1e4);
// 			Assert::AreEqual((real)-1.562500000000000e+04, fint(13), REAL_TOLERANCE*1e4);
// 			Assert::AreEqual((real)-1.281250000000000e+05, fint(14), REAL_TOLERANCE*1e5);
// 			Assert::AreEqual((real)-1.875000000000000e+04, fint(15), REAL_TOLERANCE*1e4);
// 			Assert::AreEqual((real)-2.812500000000000e+04, fint(16), REAL_TOLERANCE*1e4);
// 			Assert::AreEqual((real)-1.281250000000000e+05, fint(17), REAL_TOLERANCE*1e5);
// 			Assert::AreEqual((real)-3.125000000000000e+04, fint(18), REAL_TOLERANCE*1e4);
// 			Assert::AreEqual((real) 2.187500000000000e+04, fint(19), REAL_TOLERANCE*1e4);
// 			Assert::AreEqual((real)-1.218750000000000e+05, fint(20), REAL_TOLERANCE*1e5);
// 			Assert::AreEqual((real) 1.875000000000000e+04, fint(21), REAL_TOLERANCE*1e4);
// 			Assert::AreEqual((real) 3.437500000000000e+04, fint(22), REAL_TOLERANCE*1e4);
// 			Assert::AreEqual((real)-1.218750000000000e+05, fint(23), REAL_TOLERANCE*1e5);
// 
//       ti.Destroy();
// 		}
// 		TEST_METHOD(UpdateStressTest)
// 		{
// 			NumericalModel& model = BuildSimpleHexahedralElement();
// 			FiniteElement& fe = model.Elements().GetByInternalIndex(0);
// 			IntegrationDimension& points = fe.Geometry().GetIntegrationPoints();
// 			AnalysisTimeline& ti = AnalysisTimeline::Create(0,0,0,0);
// 
// 			fe.InitializeForAnalysis();
// 
// 			// create a spurious displacement field
// 			ColumnVector displacement(24);
// 			displacement.ClearAll();
// 			displacement(12) = 1.25e-6;
// 			displacement(13) = 2e-6;
// 			displacement(14) = -2.5e-6;
// 			displacement(15) = 1.6e-6;
// 			displacement(16) = -1.8e-6;
// 			displacement(17) = -5e-6;
// 			displacement(18) = 0;
// 			displacement(19) = -0.8e-6;
// 			displacement(20) = -2.5e-6;
// 			displacement(21) = -4e-6;
// 			displacement(22) = -1.6e-6;
// 			displacement(23) = 0.4e-6;
// 
// 			// create a spurious stress state
// 			for (int i = 0; i < points.Count(); i++)
// 			{
// 				IntegrationPoint& p = points[i];
//         p.State().Reset();
// 				Vector& strain = p.State().Strain();
// 				Vector& stress = p.State().Stress();
// 
// 				for (int j = 0; j < 6; j++)
// 				{
// 					strain(j) = 1;
// 					stress(j) = 1;
// 				}
// 			}
// 
// 			// update stress state
//       fe.UpdateStrain(displacement, model.Kinematics().Velocity(), ti);
// 			fe.UpdateStress(displacement, model.Kinematics().Velocity(), ti);
// 
// 			// Check results for integration point 0:
// 			{
// 				IntegrationPoint& p = points[0];
// 				Vector& strain = p.State().Strain();
// 				Vector& stress = p.State().Stress();
// 				Vector& strainIncr = p.State().LastStrainIncrement();
// 				Vector& stressIncr = p.State().LastStressIncrement();
// 				Assert::AreEqual((real) 2.369661282874152e-07, strainIncr(0), REAL_TOLERANCE);
// 				Assert::AreEqual((real)-5.553418012614796e-07, strainIncr(1), REAL_TOLERANCE);
// 				Assert::AreEqual((real)-2.433333333333334e-06, strainIncr(2), REAL_TOLERANCE);
// 				Assert::AreEqual((real)-1.544059892324150e-06, strainIncr(3), REAL_TOLERANCE);
// 				Assert::AreEqual((real) 1.236602540378444e-06, strainIncr(4), REAL_TOLERANCE);
// 				Assert::AreEqual((real)-1.686648580981932e-07, strainIncr(5), REAL_TOLERANCE);
// 
// 				Assert::AreEqual((real)(1 +  2.369661282874152e-07), strain(0), REAL_TOLERANCE);
// 				Assert::AreEqual((real)(1 + -5.553418012614796e-07), strain(1), REAL_TOLERANCE);
// 				Assert::AreEqual((real)(1 + -2.433333333333334e-06), strain(2), REAL_TOLERANCE);
// 				Assert::AreEqual((real)(1 + -1.544059892324150e-06), strain(3), REAL_TOLERANCE);
// 				Assert::AreEqual((real)(1 +  1.236602540378444e-06), strain(4), REAL_TOLERANCE);
// 				Assert::AreEqual((real)(1 + -1.686648580981932e-07), strain(5), REAL_TOLERANCE);
// 
// 				Assert::AreEqual((real)-2.810485579143282e+05, stressIncr(0), REAL_TOLERANCE*1E5);
// 				Assert::AreEqual((real)-4.029420855372351e+05, stressIncr(1), REAL_TOLERANCE*1E5);
// 				Assert::AreEqual((real)-6.918638597021357e+05, stressIncr(2), REAL_TOLERANCE*1E5);
// 				Assert::AreEqual((real)-1.187738378710885e+05, stressIncr(3), REAL_TOLERANCE*1E5);
// 				Assert::AreEqual((real) 9.512327233680336e+04, stressIncr(4), REAL_TOLERANCE*1E4);
// 				Assert::AreEqual((real)-1.297421985370717e+04, stressIncr(5), REAL_TOLERANCE*1E4);
// 
// 				Assert::AreEqual((real)(1 + -2.810485579143282e+05), stress(0), REAL_TOLERANCE*1E5);
// 				Assert::AreEqual((real)(1 + -4.029420855372351e+05), stress(1), REAL_TOLERANCE*1E5);
// 				Assert::AreEqual((real)(1 + -6.918638597021357e+05), stress(2), REAL_TOLERANCE*1E5);
// 				Assert::AreEqual((real)(1 + -1.187738378710885e+05), stress(3), REAL_TOLERANCE*1E5);
// 				Assert::AreEqual((real)(1 +  9.512327233680336e+04), stress(4), REAL_TOLERANCE*1E4);
// 				Assert::AreEqual((real)(1 + -1.297421985370717e+04), stress(5), REAL_TOLERANCE*1E4);
// 			}
// 
// 			// Check results for integration point 1:
// 			{
// 				IntegrationPoint& p = points[1];
// 				Vector& strain = p.State().Strain();
// 				Vector& stress = p.State().Stress();
// 				Vector& strainIncr = p.State().LastStrainIncrement();
// 				Vector& stressIncr = p.State().LastStressIncrement();
// 				Assert::AreEqual((real) 8.843696304415179e-07, strainIncr(0), REAL_TOLERANCE);
// 				Assert::AreEqual((real)-2.072563817874660e-06, strainIncr(1), REAL_TOLERANCE);
// 				Assert::AreEqual((real)-2.433333333333334e-06, strainIncr(2), REAL_TOLERANCE);
// 				Assert::AreEqual((real)-5.762509968083057e-06, strainIncr(3), REAL_TOLERANCE);
// 				Assert::AreEqual((real) 2.862114933857100e-06, strainIncr(4), REAL_TOLERANCE);
// 				Assert::AreEqual((real)-1.660843918243516e-06, strainIncr(5), REAL_TOLERANCE);
// 
// 				Assert::AreEqual((real)(1 +  8.843696304415179e-07), strain(0), REAL_TOLERANCE);
// 				Assert::AreEqual((real)(1 + -2.072563817874660e-06), strain(1), REAL_TOLERANCE);
// 				Assert::AreEqual((real)(1 + -2.433333333333334e-06), strain(2), REAL_TOLERANCE);
// 				Assert::AreEqual((real)(1 + -5.762509968083057e-06), strain(3), REAL_TOLERANCE);
// 				Assert::AreEqual((real)(1 +  2.862114933857100e-06), strain(4), REAL_TOLERANCE);
// 				Assert::AreEqual((real)(1 + -1.660843918243516e-06), strain(5), REAL_TOLERANCE);
// 
// 				Assert::AreEqual((real)-2.818116938666675e+05, stressIncr(0), REAL_TOLERANCE*1E5);
// 				Assert::AreEqual((real)-7.367245320691562e+05, stressIncr(1), REAL_TOLERANCE*1E5);
// 				Assert::AreEqual((real)-7.922275344474138e+05, stressIncr(2), REAL_TOLERANCE*1E5);
// 				Assert::AreEqual((real)-4.432699975448505e+05, stressIncr(3), REAL_TOLERANCE*1E5);
// 				Assert::AreEqual((real) 2.201626872197769e+05, stressIncr(4), REAL_TOLERANCE*1E5);
// 				Assert::AreEqual((real)-1.277572244802705e+05, stressIncr(5), REAL_TOLERANCE*1E5);
// 
// 				Assert::AreEqual((real)(1 + -2.818116938666675e+05), stress(0), REAL_TOLERANCE*1E5);
// 				Assert::AreEqual((real)(1 + -7.367245320691562e+05), stress(1), REAL_TOLERANCE*1E5);
// 				Assert::AreEqual((real)(1 + -7.922275344474138e+05), stress(2), REAL_TOLERANCE*1E5);
// 				Assert::AreEqual((real)(1 + -4.432699975448505e+05), stress(3), REAL_TOLERANCE*1E5);
// 				Assert::AreEqual((real)(1 +  2.201626872197769e+05), stress(4), REAL_TOLERANCE*1E5);
// 				Assert::AreEqual((real)(1 + -1.277572244802705e+05), stress(5), REAL_TOLERANCE*1E5);
// 			}
// 
// 			// Check results for integration point 2:
// 			{
// 				IntegrationPoint& p = points[2];
// 				Vector& strain = p.State().Strain();
// 				Vector& stress = p.State().Stress();
// 				Vector& strainIncr = p.State().LastStrainIncrement();
// 				Vector& stressIncr = p.State().LastStressIncrement();
// 				Assert::AreEqual((real) 6.822970362251488e-07, strainIncr(0), REAL_TOLERANCE);
// 				Assert::AreEqual((real)-5.553418012614796e-07, strainIncr(1), REAL_TOLERANCE);
// 				Assert::AreEqual((real)-8.078209398546773e-07, strainIncr(2), REAL_TOLERANCE);
// 				Assert::AreEqual((real)-9.828209398546772e-07, strainIncr(3), REAL_TOLERANCE);
// 				Assert::AreEqual((real)-2.806194762347363e-07, strainIncr(4), REAL_TOLERANCE);
// 				Assert::AreEqual((real)-2.803226250577253e-06, strainIncr(5), REAL_TOLERANCE);
// 
// 				Assert::AreEqual((real)(1 +  6.822970362251488e-07), strain(0), REAL_TOLERANCE);
// 				Assert::AreEqual((real)(1 + -5.553418012614796e-07), strain(1), REAL_TOLERANCE);
// 				Assert::AreEqual((real)(1 + -8.078209398546773e-07), strain(2), REAL_TOLERANCE);
// 				Assert::AreEqual((real)(1 + -9.828209398546772e-07), strain(3), REAL_TOLERANCE);
// 				Assert::AreEqual((real)(1 + -2.806194762347363e-07), strain(4), REAL_TOLERANCE);
// 				Assert::AreEqual((real)(1 + -2.803226250577253e-06), strain(5), REAL_TOLERANCE);
// 
// 				Assert::AreEqual((real) 2.640734731644503e+04, stressIncr(0), REAL_TOLERANCE*1E4);
// 				Assert::AreEqual((real)-1.639986276814978e+05, stressIncr(1), REAL_TOLERANCE*1E5);
// 				Assert::AreEqual((real)-2.028415720804513e+05, stressIncr(2), REAL_TOLERANCE*1E5);
// 				Assert::AreEqual((real)-7.560161075805209e+04, stressIncr(3), REAL_TOLERANCE*1E4);
// 				Assert::AreEqual((real)-2.158611355651818e+04, stressIncr(4), REAL_TOLERANCE*1E4);
// 				Assert::AreEqual((real)-2.156327885059425e+05, stressIncr(5), REAL_TOLERANCE*1E5);
// 
// 				Assert::AreEqual((real)(1 +  2.640734731644503e+04), stress(0), REAL_TOLERANCE*1E4);
// 				Assert::AreEqual((real)(1 + -1.639986276814978e+05), stress(1), REAL_TOLERANCE*1E5);
// 				Assert::AreEqual((real)(1 + -2.028415720804513e+05), stress(2), REAL_TOLERANCE*1E5);
// 				Assert::AreEqual((real)(1 + -7.560161075805209e+04), stress(3), REAL_TOLERANCE*1E4);
// 				Assert::AreEqual((real)(1 + -2.158611355651818e+04), stress(4), REAL_TOLERANCE*1E4);
// 				Assert::AreEqual((real)(1 + -2.156327885059425e+05), stress(5), REAL_TOLERANCE*1E5);
// 			}
// 
// 			// Check results for integration point 3:
// 			{
// 				IntegrationPoint& p = points[3];
// 				Vector& strain = p.State().Strain();
// 				Vector& stress = p.State().Stress();
// 				Vector& strainIncr = p.State().LastStrainIncrement();
// 				Vector& stressIncr = p.State().LastStressIncrement();
// 				Assert::AreEqual((real) 2.546367205045918e-06, strainIncr(0), REAL_TOLERANCE);
// 				Assert::AreEqual((real)-2.072563817874660e-06, strainIncr(1), REAL_TOLERANCE);
// 				Assert::AreEqual((real)-8.078209398546773e-07, strainIncr(2), REAL_TOLERANCE);
// 				Assert::AreEqual((real)-3.667937682280251e-06, strainIncr(3), REAL_TOLERANCE);
// 				Assert::AreEqual((real) 1.344892917243919e-06, strainIncr(4), REAL_TOLERANCE);
// 				Assert::AreEqual((real)-4.428738644055910e-06, strainIncr(5), REAL_TOLERANCE);
// 
// 				Assert::AreEqual((real)(1 +  2.546367205045918e-06), strain(0), REAL_TOLERANCE);
// 				Assert::AreEqual((real)(1 + -2.072563817874660e-06), strain(1), REAL_TOLERANCE);
// 				Assert::AreEqual((real)(1 + -8.078209398546773e-07), strain(2), REAL_TOLERANCE);
// 				Assert::AreEqual((real)(1 + -3.667937682280251e-06), strain(3), REAL_TOLERANCE);
// 				Assert::AreEqual((real)(1 +  1.344892917243919e-06), strain(4), REAL_TOLERANCE);
// 				Assert::AreEqual((real)(1 + -4.428738644055910e-06), strain(5), REAL_TOLERANCE);
// 
// 				Assert::AreEqual((real) 3.532083139282083e+05, stressIncr(0), REAL_TOLERANCE*1E5);
// 				Assert::AreEqual((real)-3.573964588288036e+05, stressIncr(1), REAL_TOLERANCE*1E5);
// 				Assert::AreEqual((real)-1.628206314411140e+05, stressIncr(2), REAL_TOLERANCE*1E5);
// 				Assert::AreEqual((real)-2.821490524830962e+05, stressIncr(3), REAL_TOLERANCE*1E5);
// 				Assert::AreEqual((real) 1.034533013264553e+05, stressIncr(4), REAL_TOLERANCE*1E5);
// 				Assert::AreEqual((real)-3.406722033889161e+05, stressIncr(5), REAL_TOLERANCE*1E5);
// 
// 				Assert::AreEqual((real)(1 +  3.532083139282083e+05), stress(0), REAL_TOLERANCE*1E5);
// 				Assert::AreEqual((real)(1 + -3.573964588288036e+05), stress(1), REAL_TOLERANCE*1E5);
// 				Assert::AreEqual((real)(1 + -1.628206314411140e+05), stress(2), REAL_TOLERANCE*1E5);
// 				Assert::AreEqual((real)(1 + -2.821490524830962e+05), stress(3), REAL_TOLERANCE*1E5);
// 				Assert::AreEqual((real)(1 +  1.034533013264553e+05), stress(4), REAL_TOLERANCE*1E5);
// 				Assert::AreEqual((real)(1 + -3.406722033889161e+05), stress(5), REAL_TOLERANCE*1E5);
// 			}
// 
// 			// Check results for integration point 4:
// 			{
// 				IntegrationPoint& p = points[4];
// 				Vector& strain = p.State().Strain();
// 				Vector& stress = p.State().Stress();
// 				Vector& strainIncr = p.State().LastStrainIncrement();
// 				Vector& stressIncr = p.State().LastStressIncrement();
// 				Assert::AreEqual((real) 2.369661282874152e-07, strainIncr(0), REAL_TOLERANCE);
// 				Assert::AreEqual((real) 5.897151207993051e-09, strainIncr(1), REAL_TOLERANCE);
// 				Assert::AreEqual((real)-3.925512393478656e-06, strainIncr(2), REAL_TOLERANCE);
// 				Assert::AreEqual((real)-1.098728984386416e-06, strainIncr(3), REAL_TOLERANCE);
// 				Assert::AreEqual((real)-4.448929172439199e-07, strainIncr(4), REAL_TOLERANCE);
// 				Assert::AreEqual((real) 4.787386440559094e-07, strainIncr(5), REAL_TOLERANCE);
// 
// 				Assert::AreEqual((real)(1 +  2.369661282874152e-07), strain(0), REAL_TOLERANCE);
// 				Assert::AreEqual((real)(1 +  5.897151207993051e-09), strain(1), REAL_TOLERANCE);
// 				Assert::AreEqual((real)(1 + -3.925512393478656e-06), strain(2), REAL_TOLERANCE);
// 				Assert::AreEqual((real)(1 + -1.098728984386416e-06), strain(3), REAL_TOLERANCE);
// 				Assert::AreEqual((real)(1 + -4.448929172439199e-07), strain(4), REAL_TOLERANCE);
// 				Assert::AreEqual((real)(1 +  4.787386440559094e-07), strain(5), REAL_TOLERANCE);
// 
// 				Assert::AreEqual((real)-3.884647241846186e+05, stressIncr(0), REAL_TOLERANCE*1E5);
// 				Assert::AreEqual((real)-4.240137975814528e+05, stressIncr(1), REAL_TOLERANCE*1E5);
// 				Assert::AreEqual((real)-1.028846035225553e+06, stressIncr(2), REAL_TOLERANCE*1E6);
// 				Assert::AreEqual((real)-8.451761418357046e+04, stressIncr(3), REAL_TOLERANCE*1E4);
// 				Assert::AreEqual((real)-3.422253209568614e+04, stressIncr(4), REAL_TOLERANCE*1E4);
// 				Assert::AreEqual((real) 3.682604954276226e+04, stressIncr(5), REAL_TOLERANCE*1E4);
// 
// 				Assert::AreEqual((real)(1 + -3.884647241846186e+05), stress(0), REAL_TOLERANCE*1E5);
// 				Assert::AreEqual((real)(1 + -4.240137975814528e+05), stress(1), REAL_TOLERANCE*1E5);
// 				Assert::AreEqual((real)(1 + -1.028846035225553e+06), stress(2), REAL_TOLERANCE*1E6);
// 				Assert::AreEqual((real)(1 + -8.451761418357046e+04), stress(3), REAL_TOLERANCE*1E4);
// 				Assert::AreEqual((real)(1 + -3.422253209568614e+04), stress(4), REAL_TOLERANCE*1E4);
// 				Assert::AreEqual((real)(1 +  3.682604954276226e+04), stress(5), REAL_TOLERANCE*1E4);
// 			}
// 
// 			// Check results for integration point 5:
// 			{
// 				IntegrationPoint& p = points[5];
// 				Vector& strain = p.State().Strain();
// 				Vector& stress = p.State().Stress();
// 				Vector& strainIncr = p.State().LastStrainIncrement();
// 				Vector& stressIncr = p.State().LastStressIncrement();
// 				Assert::AreEqual((real) 8.843696304415179e-07, strainIncr(0), REAL_TOLERANCE);
// 				Assert::AreEqual((real) 2.200846792814611e-08, strainIncr(1), REAL_TOLERANCE);
// 				Assert::AreEqual((real)-3.925512393478656e-06, strainIncr(2), REAL_TOLERANCE);
// 				Assert::AreEqual((real)-4.100512393478656e-06, strainIncr(3), REAL_TOLERANCE);
// 				Assert::AreEqual((real) 1.047286142901403e-06, strainIncr(4), REAL_TOLERANCE);
// 				Assert::AreEqual((real)-1.013440416089413e-06, strainIncr(5), REAL_TOLERANCE);
// 
// 				Assert::AreEqual((real)(1 +  8.843696304415179e-07), strain(0), REAL_TOLERANCE);
// 				Assert::AreEqual((real)(1 +  2.200846792814611e-08), strain(1), REAL_TOLERANCE);
// 				Assert::AreEqual((real)(1 + -3.925512393478656e-06), strain(2), REAL_TOLERANCE);
// 				Assert::AreEqual((real)(1 + -4.100512393478656e-06), strain(3), REAL_TOLERANCE);
// 				Assert::AreEqual((real)(1 +  1.047286142901403e-06), strain(4), REAL_TOLERANCE);
// 				Assert::AreEqual((real)(1 + -1.013440416089413e-06), strain(5), REAL_TOLERANCE);
// 
// 				Assert::AreEqual((real)-2.123047832138810e+05, stressIncr(0), REAL_TOLERANCE*1E5);
// 				Assert::AreEqual((real)-3.449757312928612e+05, stressIncr(1), REAL_TOLERANCE*1E5);
// 				Assert::AreEqual((real)-9.522866330477539e+05, stressIncr(2), REAL_TOLERANCE*1E5);
// 				Assert::AreEqual((real)-3.154240302675889e+05, stressIncr(3), REAL_TOLERANCE*1E5);
// 				Assert::AreEqual((real) 8.056047253087712e+04, stressIncr(4), REAL_TOLERANCE*1E5);
// 				Assert::AreEqual((real)-7.795695508380103e+04, stressIncr(5), REAL_TOLERANCE*1E5);
// 
// 				Assert::AreEqual((real)(1 + -2.123047832138810e+05), stress(0), REAL_TOLERANCE*1E5);
// 				Assert::AreEqual((real)(1 + -3.449757312928612e+05), stress(1), REAL_TOLERANCE*1E5);
// 				Assert::AreEqual((real)(1 + -9.522866330477539e+05), stress(2), REAL_TOLERANCE*1E5);
// 				Assert::AreEqual((real)(1 + -3.154240302675889e+05), stress(3), REAL_TOLERANCE*1E5);
// 				Assert::AreEqual((real)(1 +  8.056047253087712e+04), stress(4), REAL_TOLERANCE*1E5);
// 				Assert::AreEqual((real)(1 + -7.795695508380103e+04), stress(5), REAL_TOLERANCE*1E5);
// 			}
// 
// 			// Check results for integration point 6:
// 			{
// 				IntegrationPoint& p = points[6];
// 				Vector& strain = p.State().Strain();
// 				Vector& stress = p.State().Stress();
// 				Vector& strainIncr = p.State().LastStrainIncrement();
// 				Vector& stressIncr = p.State().LastStressIncrement();
// 				Assert::AreEqual((real) 6.822970362251488e-07, strainIncr(0), REAL_TOLERANCE);
// 				Assert::AreEqual((real) 5.897151207993051e-09, strainIncr(1), REAL_TOLERANCE);
// 				Assert::AreEqual((real)-2.433333333333333e-06, strainIncr(2), REAL_TOLERANCE);
// 				Assert::AreEqual((real)-5.374900319169435e-07, strainIncr(3), REAL_TOLERANCE);
// 				Assert::AreEqual((real)-4.287816005237668e-07, strainIncr(4), REAL_TOLERANCE);
// 				Assert::AreEqual((real)-9.391560817564838e-07, strainIncr(5), REAL_TOLERANCE);
// 
// 				Assert::AreEqual((real)(1 +  6.822970362251488e-07), strain(0), REAL_TOLERANCE);
// 				Assert::AreEqual((real)(1 +  5.897151207993051e-09), strain(1), REAL_TOLERANCE);
// 				Assert::AreEqual((real)(1 + -2.433333333333333e-06), strain(2), REAL_TOLERANCE);
// 				Assert::AreEqual((real)(1 + -5.374900319169435e-07), strain(3), REAL_TOLERANCE);
// 				Assert::AreEqual((real)(1 + -4.287816005237668e-07), strain(4), REAL_TOLERANCE);
// 				Assert::AreEqual((real)(1 + -9.391560817564838e-07), strain(5), REAL_TOLERANCE);
// 
// 				Assert::AreEqual((real)-9.639343433846074e+04, stressIncr(0), REAL_TOLERANCE*1E4);
// 				Assert::AreEqual((real)-2.004549551103308e+05, stressIncr(1), REAL_TOLERANCE*1E5);
// 				Assert::AreEqual((real)-5.757211835013040e+05, stressIncr(2), REAL_TOLERANCE*1E5);
// 				Assert::AreEqual((real)-4.134538707053412e+04, stressIncr(3), REAL_TOLERANCE*1E4);
// 				Assert::AreEqual((real)-3.298320004028975e+04, stressIncr(4), REAL_TOLERANCE*1E4);
// 				Assert::AreEqual((real)-7.224277551972952e+04, stressIncr(5), REAL_TOLERANCE*1E4);
// 
// 				Assert::AreEqual((real)(1 + -9.639343433846074e+04), stress(0), REAL_TOLERANCE*1E4);
// 				Assert::AreEqual((real)(1 + -2.004549551103308e+05), stress(1), REAL_TOLERANCE*1E5);
// 				Assert::AreEqual((real)(1 + -5.757211835013040e+05), stress(2), REAL_TOLERANCE*1E5);
// 				Assert::AreEqual((real)(1 + -4.134538707053412e+04), stress(3), REAL_TOLERANCE*1E4);
// 				Assert::AreEqual((real)(1 + -3.298320004028975e+04), stress(4), REAL_TOLERANCE*1E4);
// 				Assert::AreEqual((real)(1 + -7.224277551972952e+04), stress(5), REAL_TOLERANCE*1E4);
// 			}
// 
// 			// Check results for integration point 7:
// 			{
// 				IntegrationPoint& p = points[7];
// 				Vector& strain = p.State().Strain();
// 				Vector& stress = p.State().Stress();
// 				Vector& strainIncr = p.State().LastStrainIncrement();
// 				Vector& stressIncr = p.State().LastStressIncrement();
// 				Assert::AreEqual((real) 2.546367205045918e-06, strainIncr(0), REAL_TOLERANCE);
// 				Assert::AreEqual((real) 2.200846792814611e-08, strainIncr(1), REAL_TOLERANCE);
// 				Assert::AreEqual((real)-2.433333333333333e-06, strainIncr(2), REAL_TOLERANCE);
// 				Assert::AreEqual((real)-2.005940107675850e-06, strainIncr(3), REAL_TOLERANCE);
// 				Assert::AreEqual((real) 1.063397459621556e-06, strainIncr(4), REAL_TOLERANCE);
// 				Assert::AreEqual((real)-2.564668475235140e-06, strainIncr(5), REAL_TOLERANCE);
// 
// 				Assert::AreEqual((real)(1 +  2.546367205045918e-06), strain(0), REAL_TOLERANCE);
// 				Assert::AreEqual((real)(1 +  2.200846792814611e-08), strain(1), REAL_TOLERANCE);
// 				Assert::AreEqual((real)(1 + -2.433333333333333e-06), strain(2), REAL_TOLERANCE);
// 				Assert::AreEqual((real)(1 + -2.005940107675850e-06), strain(3), REAL_TOLERANCE);
// 				Assert::AreEqual((real)(1 +  1.063397459621556e-06), strain(4), REAL_TOLERANCE);
// 				Assert::AreEqual((real)(1 + -2.564668475235140e-06), strain(5), REAL_TOLERANCE);
// 
// 				Assert::AreEqual((real) 4.073306091963794e+05, stressIncr(0), REAL_TOLERANCE*1E5);
// 				Assert::AreEqual((real) 1.896772656287608e+04, stressIncr(1), REAL_TOLERANCE*1E4);
// 				Assert::AreEqual((real)-3.587771659388899e+05, stressIncr(2), REAL_TOLERANCE*1E5);
// 				Assert::AreEqual((real)-1.543030852058346e+05, stressIncr(3), REAL_TOLERANCE*1E5);
// 				Assert::AreEqual((real) 8.179980458627353e+04, stressIncr(4), REAL_TOLERANCE*1E4);
// 				Assert::AreEqual((real)-1.972821904027030e+05, stressIncr(5), REAL_TOLERANCE*1E5);
// 
// 				Assert::AreEqual((real)(1 +  4.073306091963794e+05), stress(0), REAL_TOLERANCE*1E5);
// 				Assert::AreEqual((real)(1 +  1.896772656287608e+04), stress(1), REAL_TOLERANCE*1E4);
// 				Assert::AreEqual((real)(1 + -3.587771659388899e+05), stress(2), REAL_TOLERANCE*1E5);
// 				Assert::AreEqual((real)(1 + -1.543030852058346e+05), stress(3), REAL_TOLERANCE*1E5);
// 				Assert::AreEqual((real)(1 +  8.179980458627353e+04), stress(4), REAL_TOLERANCE*1E4);
// 				Assert::AreEqual((real)(1 + -1.972821904027030e+05), stress(5), REAL_TOLERANCE*1E5);
// 			}
// 			
// 
// 			// element stress must also have been updated
//       Vector& strain = fe.PhysicalState().Strain();
//       Vector& stress = fe.PhysicalState().Stress();
// 			Assert::AreEqual((real)(1 +  1.087500000000000e-06), strain(0), REAL_TOLERANCE);
// 			Assert::AreEqual((real)(1 + -6.500000000000001e-07), strain(1), REAL_TOLERANCE);
// 			Assert::AreEqual((real)(1 + -2.400000000000000e-06), strain(2), REAL_TOLERANCE);
// 			Assert::AreEqual((real)(1 + -2.462500000000000e-06), strain(3), REAL_TOLERANCE);
// 			Assert::AreEqual((real)(1 +  7.999999999999999e-07), strain(4), REAL_TOLERANCE);
// 			Assert::AreEqual((real)(1 + -1.637500000000000e-06), strain(5), REAL_TOLERANCE);
// 
// 			Assert::AreEqual((real)(1 + -5.913461538461538e+04), stress(0), REAL_TOLERANCE*1E4);
// 			Assert::AreEqual((real)(1 + -3.264423076923078e+05), stress(1), REAL_TOLERANCE*1E5);
// 			Assert::AreEqual((real)(1 + -5.956730769230769e+05), stress(2), REAL_TOLERANCE*1E5);
// 			Assert::AreEqual((real)(1 + -1.894230769230769e+05), stress(3), REAL_TOLERANCE*1E5);
// 			Assert::AreEqual((real)(1 +  6.153846153846152e+04), stress(4), REAL_TOLERANCE*1E4);
// 			Assert::AreEqual((real)(1 + -1.259615384615385e+05), stress(5), REAL_TOLERANCE*1E5);
// 
// 			ti.Destroy();
// 		}
// 	};
// 
// 
// } } }
// 
// #endif
