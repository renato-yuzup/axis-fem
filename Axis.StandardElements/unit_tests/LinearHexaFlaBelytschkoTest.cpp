#if defined _DEBUG || defined DEBUG
// #include "unit_tests.hpp"
// #include "domain/analyses/AnalysisTimeline.hpp"
// #include "domain/elements/FiniteElement.hpp"
// #include "domain/elements/Node.hpp"
// #include "domain/elements/ElementGeometry.hpp"
// #include "domain/integration/IntegrationDimension.hpp"
// #include "domain/formulations/LinearHexaFlanaganBelytschkoFormulation.hpp"
// #include "domain/materials/MaterialModel.hpp"
// #include "domain/physics/InfinitesimalState.hpp"
// #include "MockLinearMaterial.hpp"
// #include "foundation/blas/VectorAlgebra.hpp"
// 
// namespace ada = axis::domain::analyses;
// namespace ade = axis::domain::elements;
// namespace adf = axis::domain::formulations;
// namespace adi = axis::domain::integration;
// namespace adm = axis::domain::materials;
// namespace afb = axis::foundation::blas;
// 
// namespace axis { namespace unit_tests { namespace standard_elements {
// 
// /*
//   Tests Linear Flanagan-Belytschko (1981) formulation for hexahedral elements.
// **/
// TEST_CLASS(LinearHexaFlaBelytschkoTest)
// {
// public:
//   ade::FiniteElement &CreateTestElement(real poisson)
//   {
//     ade::Node &n1 = *new ade::Node(0, 0, 0,0,0);
//     ade::Node &n2 = *new ade::Node(1, 1, 1,0,0);
//     ade::Node &n3 = *new ade::Node(2, 2, 1,1,0);
//     ade::Node &n4 = *new ade::Node(3, 3, 0,1,0);
//     ade::Node &n5 = *new ade::Node(4, 4, 0,0,1);
//     ade::Node &n6 = *new ade::Node(5, 5, 1,0,1);
//     ade::Node &n7 = *new ade::Node(6, 6, 1,1,1);
//     ade::Node &n8 = *new ade::Node(7, 7, 0,1,1);
// 
//     adi::IntegrationDimension& set = *new adi::IntegrationDimension(1);
//     adi::IntegrationPoint& p = *new adi::IntegrationPoint(0,0,0,1);
//     p.State().Reset();
//     set.AddIntegrationPoint(p);
// 
//     ade::ElementGeometry &geometry = *new ade::ElementGeometry(8,6,12);
//     geometry.SetNode(0, n1);
//     geometry.SetNode(1, n2);
//     geometry.SetNode(2, n3);
//     geometry.SetNode(3, n4);
//     geometry.SetNode(4, n5);
//     geometry.SetNode(5, n6);
//     geometry.SetNode(6, n7);
//     geometry.SetNode(7, n8);
// 
//     adm::MaterialModel& material = *new MockLinearMaterial(poisson);
//     adf::LinearHexaFlanaganBelytschkoFormulation &formulation = 
//       *new adf::LinearHexaFlanaganBelytschkoFormulation(0.1);
//     ade::FiniteElement &hexahedronElement = *new ade::FiniteElement(1, geometry, material, formulation);
// 
//     n1.ConnectElement(hexahedronElement);
//     n2.ConnectElement(hexahedronElement);
//     n3.ConnectElement(hexahedronElement);
//     n4.ConnectElement(hexahedronElement);
//     n5.ConnectElement(hexahedronElement);
//     n6.ConnectElement(hexahedronElement);
//     n7.ConnectElement(hexahedronElement);
//     n8.ConnectElement(hexahedronElement);
// 
//     hexahedronElement.InitializeForAnalysis();
//     hexahedronElement.InitializeForStep();
//     return hexahedronElement;
//   }
// 
//   TEST_METHOD(TestInternalForceSimpleCaseNoPoisson)
//   {
//     ada::AnalysisTimeline& tm = ada::AnalysisTimeline::Create(0, 1, 0.2, 1e-6);
//     ade::FiniteElement& element = CreateTestElement(0);
//     afb::ColumnVector du(24), v(24), fint(24);
//     du.ClearAll();
//     for (int i = 14; i < 24; i += 3)
//     {
//       du(i) = -2.5e-7;
//     }
//     v.ClearAll();
//     v.CopyFrom(du);
//     v.Scale(1 / 1e-6);
// 
//     element.ExtractInternalForce(fint, du, du, v, tm);
//     for (int i = 0; i < 24; ++i)
//     {
//       if (i > 12 && (i-2) % 3 == 0)
//       {
//         Assert::AreEqual(5e4, fint(i), 1e-2);
//       }
//       else
//       {
//         Assert::AreEqual(0, fint(i), 1e-2);
//       }
//     }
//     tm.Destroy();
//     element.Destroy();
//   }
// };
// 
// } } } // namespace axis::unit_tests::standard_elements

#endif