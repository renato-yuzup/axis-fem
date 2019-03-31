#if defined _DEBUG || defined DEBUG
#include "unit_tests.hpp"
#include "domain/analyses/AnalysisTimeline.hpp"
#include "domain/elements/FiniteElement.hpp"
#include "domain/elements/Node.hpp"
#include "domain/elements/ElementGeometry.hpp"
#include "domain/elements/MatrixOption.hpp"
#include "domain/formulations/NonLinearHexaReducedFormulation.hpp"
#include "domain/materials/MaterialModel.hpp"
#include "domain/physics/InfinitesimalState.hpp"
#include "MockLinearMaterial.hpp" // we won't actually use it, so it's okay

namespace ada = axis::domain::analyses;
namespace ade = axis::domain::elements;
namespace adf = axis::domain::formulations;
namespace adi = axis::domain::integration;
namespace adm = axis::domain::materials;
namespace afb = axis::foundation::blas;
namespace afm = axis::foundation::memory;

namespace axis { namespace unit_tests { namespace standard_elements {

/*
  Tests non-linear, first-order, hexahedral reduced integration formulation.
**/
TEST_CLASS(NonLinearHexaReducedIntegrationTestFixture)
{
public:
  TEST_CLASS_INITIALIZE(SetUp)
  {
    // initialize memory arenas
    axis::System::Initialize();
  }
  TEST_CLASS_CLEANUP(TearDown)
  {
    // destroy memory arenas
    axis::System::Finalize();
  }

  ade::FiniteElement &CreateTestElement(real poisson)
  {
    // create nodes
    afm::RelativePointer n1 = ade::Node::Create(0, 0, 0,0,0);
    afm::RelativePointer n2 = ade::Node::Create(1, 1, 1,0,0);
    afm::RelativePointer n3 = ade::Node::Create(2, 2, 1,1,0);
    afm::RelativePointer n4 = ade::Node::Create(3, 3, 0,1,0);
    afm::RelativePointer n5 = ade::Node::Create(4, 4, 0,0,1);
    afm::RelativePointer n6 = ade::Node::Create(5, 5, 1,0,1);
    afm::RelativePointer n7 = ade::Node::Create(6, 6, 1,1,1);
    afm::RelativePointer n8 = ade::Node::Create(7, 7, 0,1,1);
    ade::Node &node1 = absref<ade::Node>(n1);
    ade::Node &node2 = absref<ade::Node>(n2);
    ade::Node &node3 = absref<ade::Node>(n3);
    ade::Node &node4 = absref<ade::Node>(n4);
    ade::Node &node5 = absref<ade::Node>(n5);
    ade::Node &node6 = absref<ade::Node>(n6);
    ade::Node &node7 = absref<ade::Node>(n7);
    ade::Node &node8 = absref<ade::Node>(n8);

    // assign them to a geometry
    afm::RelativePointer geomPtr = ade::ElementGeometry::Create(8, 1);
    ade::ElementGeometry &geometry = absref<ade::ElementGeometry>(geomPtr);
    geometry.SetNode(0, n1); geometry.SetNode(1, n2);
    geometry.SetNode(2, n3); geometry.SetNode(3, n4);
    geometry.SetNode(4, n5); geometry.SetNode(5, n6);
    geometry.SetNode(6, n7); geometry.SetNode(7, n8);

    // set integration points
    afm::RelativePointer p = adi::IntegrationPoint::Create(0,0,0, 1);
    geometry.SetIntegrationPoint(0, p);

    adm::MaterialModel& material = *new MockLinearMaterial(poisson);
    adf::NonLinearHexaReducedFormulation &formulation = 
      *new adf::NonLinearHexaReducedFormulation();
    afm::RelativePointer hexahedronElement = 
      ade::FiniteElement::Create(1, geomPtr, material, formulation);

    node1.ConnectElement(hexahedronElement);
    node2.ConnectElement(hexahedronElement);
    node3.ConnectElement(hexahedronElement);
    node4.ConnectElement(hexahedronElement);
    node5.ConnectElement(hexahedronElement);
    node6.ConnectElement(hexahedronElement);
    node7.ConnectElement(hexahedronElement);
    node8.ConnectElement(hexahedronElement);

    ade::FiniteElement& element = absref<ade::FiniteElement>(hexahedronElement);
    element.AllocateMemory();
    element.CalculateInitialState();
    return element;
  }

  void ApplyDisplacement(ade::FiniteElement& fe, const afb::ColumnVector& u)
  {
    ade::ElementGeometry& g = fe.Geometry();
    for (int i = 0; i < g.GetNodeCount(); i++)
    {
      ade::Node& node = g[i];
      id_type id = node.GetInternalId();
      node.CurrentX() = node.X() + u(3*id + 0);
      node.CurrentY() = node.Y() + u(3*id + 1);
      node.CurrentZ() = node.Z() + u(3*id + 2);
    }
  }

  TEST_METHOD(TestDeformationGradient)
  {
    /*
        This test is based on the Example 9.1 by Bhatti, M.A. (2006): Advanced
        Topics in Finite Element Analysis of Structures, Wiley, pp. 470-471.
    */
    const real test_tolerance = 1e-5;
    ada::AnalysisTimeline& timeline = ada::AnalysisTimeline::Create(0, 1, 0, 0);
    afb::ColumnVector du(24), dv(24);
    du(0)  = 2;           du(1)  = 1;           du(2)  = 2;
    du(3)  = 2.039230;    du(4)  = 0.4;         du(5)  = 2;
    du(6)  = 2.639230;    du(7)  = 0.439230;    du(8)  = 2;
    du(9)  = 2.6;         du(10) = 1.039230;    du(11) = 2;
    du(12) = 2;           du(13) = 1;           du(14) = 2.2;
    du(15) = 2.039230;    du(16) = 0.4;         du(17) = 2.2;
    du(18) = 2.639230;    du(19) = 0.439230;    du(20) = 2.2;
    du(21) = 2.6;         du(22) = 1.039230;    du(23) = 2.2;
    dv.ClearAll();

    ade::FiniteElement& e = CreateTestElement(0.3);
    ApplyDisplacement(e, du);
    
    // this will trigger deformation gradient (F) update
    e.UpdateStrain(du);

    afb::DenseMatrix& F = e.PhysicalState().DeformationGradient();
    Assert::AreEqual((real) 1.03923, F(0,0), test_tolerance);
    Assert::AreEqual((real) 0.60000, F(0,1), test_tolerance);
    Assert::AreEqual((real) 0.00000, F(0,2), test_tolerance);

    Assert::AreEqual((real)-0.60000, F(1,0), test_tolerance);
    Assert::AreEqual((real) 1.03923, F(1,1), test_tolerance);
    Assert::AreEqual((real) 0.00000, F(1,2), test_tolerance);

    Assert::AreEqual((real) 0.00000, F(2,0), test_tolerance);
    Assert::AreEqual((real) 0.00000, F(2,1), test_tolerance);
    Assert::AreEqual((real) 1.20000, F(2,2), test_tolerance);
  }
};

} } } // namespace axis::unit_tests::standard_elements

#endif