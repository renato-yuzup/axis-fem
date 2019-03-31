#if defined _DEBUG || defined DEBUG

#include "unit_tests.hpp"
#include "domain/analyses/AnalysisTimeline.hpp"
#include "domain/integration/IntegrationPoint.hpp"
#include "domain/materials/LinearElasticIsotropicModel.hpp"
#include "domain/physics/InfinitesimalState.hpp"
#include "domain/physics/UpdatedPhysicalState.hpp"
#include "foundation/blas/DenseMatrix.hpp"
#include "foundation/blas/ColumnVector.hpp"
#include "foundation/memory/pointer.hpp"
#include "System.hpp"

namespace ada = axis::domain::analyses;
namespace adi = axis::domain::integration;
namespace adm = axis::domain::materials;
namespace adp = axis::domain::physics;
namespace afb = axis::foundation::blas;
namespace afm = axis::foundation::memory;

namespace axis { namespace unit_tests { namespace standard_materials {

TEST_CLASS(LinearElasticIsotropicModelTest)
{
  TEST_METHOD_INITIALIZE(SetUp)
  {
    axis::System::Initialize();
  }

  TEST_METHOD_CLEANUP(TearDown)
  {
    axis::System::Finalize();
  }

	TEST_METHOD(ConstructorTest)
	{
		adm::LinearElasticIsotropicModel mat(200.0e9, 0.3, 7850.0);
			
		// ok, nothing more to do, just to check it builds ok
	}
	TEST_METHOD(GetMaterialTensorTest)
	{
		real E = 200e9;
		real nu = 0.3;
		real rho = 7850;

		adm::LinearElasticIsotropicModel mat(E, nu, rho);

		const afb::DenseMatrix& matTensor = mat.GetMaterialTensor();
			
		real c11 = E*(1-nu) / ((1+nu)*(1-2*nu));
		real c12 = E*nu / ((1+nu)*(1-2*nu));
		real g = E / (2*(1+nu));

		Assert::AreEqual(c11, matTensor(0,0), REAL_ROUNDOFF_TOLERANCE);
		Assert::AreEqual(c11, matTensor(1,1), REAL_ROUNDOFF_TOLERANCE);
		Assert::AreEqual(c11, matTensor(2,2), REAL_ROUNDOFF_TOLERANCE);
		Assert::AreEqual(g,   matTensor(3,3), REAL_ROUNDOFF_TOLERANCE);
		Assert::AreEqual(g,   matTensor(4,4), REAL_ROUNDOFF_TOLERANCE);
		Assert::AreEqual(g,   matTensor(5,5), REAL_ROUNDOFF_TOLERANCE);
		Assert::AreEqual(c12, matTensor(0,1), REAL_ROUNDOFF_TOLERANCE);
		Assert::AreEqual(c12, matTensor(0,2), REAL_ROUNDOFF_TOLERANCE);
		Assert::AreEqual(c12, matTensor(1,2), REAL_ROUNDOFF_TOLERANCE);
		Assert::AreEqual(c12, matTensor(1,0), REAL_ROUNDOFF_TOLERANCE);
		Assert::AreEqual(c12, matTensor(2,0), REAL_ROUNDOFF_TOLERANCE);
		Assert::AreEqual(c12, matTensor(2,1), REAL_ROUNDOFF_TOLERANCE);
		for (int i = 3; i < 6; i++)
		{
			for (int j = 0; j < 3; j++)
			{
				Assert::AreEqual(0,   matTensor(j,i), REAL_TOLERANCE);
				Assert::AreEqual(0,   matTensor(j+3,i-3), REAL_TOLERANCE);
			}
		}
		Assert::AreEqual(0,   matTensor(3,4), REAL_TOLERANCE);
		Assert::AreEqual(0,   matTensor(3,5), REAL_TOLERANCE);
		Assert::AreEqual(0,   matTensor(4,3), REAL_TOLERANCE);
		Assert::AreEqual(0,   matTensor(4,5), REAL_TOLERANCE);
		Assert::AreEqual(0,   matTensor(5,3), REAL_TOLERANCE);
		Assert::AreEqual(0,   matTensor(5,4), REAL_TOLERANCE);
	}
	TEST_METHOD(UpdateStressTest)
	{
		real E = 200e9;
		real nu = 0.3;
		real rho = 7850;

		adm::LinearElasticIsotropicModel mat(E, nu, rho);
		ada::AnalysisTimeline& ti = ada::AnalysisTimeline::Create(0, 0, 0, 0);

    afm::RelativePointer ptr = adi::IntegrationPoint::Create(0,0,0, 1);
		adi::IntegrationPoint& p = absref<adi::IntegrationPoint>(ptr);
		p.State().Reset();

		// introduce a spurious strain state
		afb::ColumnVector& dstrain = p.State().LastStrainIncrement();
		dstrain(0) = 1.5e-6;
		dstrain(1) = -5.0e-6;
		dstrain(2) = 1.5e-6;
		dstrain(3) = +4.5e-8;
		dstrain(4) = -4.5e-8;
		dstrain(5) = -4.5e-8;

		// introduce a spurious previous stress state
		afb::ColumnVector& stress = p.State().Stress();
		stress(0) = 1.0e4;
		stress(1) = 2.0e4;
		stress(2) = 3.0e4;
		stress(3) = 4.0e4;
		stress(4) = 5.0e4;
		stress(5) = 6.0e4;
			
    adp::UpdatedPhysicalState ups(p.State());
		mat.UpdateStresses(ups, p.State(), ti, 0);
		/*
		1.0e+05 *

			0.1000   -9.8000    0.3000    0.4346    0.4654    0.5654
		*/
		Assert::AreEqual((real)9999.99999999997, stress(0), (real)(REAL_ROUNDOFF_TOLERANCE*1E3));
		Assert::AreEqual((real)-980000,          stress(1), (real)(REAL_ROUNDOFF_TOLERANCE*1E5));
		Assert::AreEqual((real)30000,            stress(2), (real)(REAL_ROUNDOFF_TOLERANCE*1E4));
		Assert::AreEqual((real)43461.5384615385, stress(3), (real)(REAL_ROUNDOFF_TOLERANCE*1E4));
		Assert::AreEqual((real)46538.4615384615, stress(4), (real)(REAL_ROUNDOFF_TOLERANCE*1E4));
		Assert::AreEqual((real)56538.4615384615, stress(5), (real)(REAL_ROUNDOFF_TOLERANCE*1E4));
	}
	TEST_METHOD(GetWavePropagationSpeedTest)
	{
		real E = 200e9;
		real nu = 0.3;
		real rho = 7850;

		adm::LinearElasticIsotropicModel mat(E, nu, rho);

		// TODO : implement this test!!
	}
};

} } }

#endif
