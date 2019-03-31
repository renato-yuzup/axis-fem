#include "LinearElasticIsotropicModel.hpp"
#include "LinearElasticIsotropicStrategy.hpp"
#include "domain/physics/UpdatedPhysicalState.hpp"
#include "domain/physics/InfinitesimalState.hpp"
#include "foundation/blas/blas.hpp"
#include "System.hpp"
#include "foundation/memory/HeapBlockArena.hpp"
#include "foundation/memory/pointer.hpp"

#define S(i, j, x)		D.SetElement(i, j, x);

namespace ada = axis::domain::analyses;
namespace adm = axis::domain::materials;
namespace adp = axis::domain::physics;
namespace afb = axis::foundation::blas;
namespace afm = axis::foundation::memory;
namespace afu = axis::foundation::uuids;

struct GPUMaterial
{
  real YoungModulus;
  real PoissonCoefficient;
};

adm::LinearElasticIsotropicModel::LinearElasticIsotropicModel( 
  real youngModulus, real poissonCoefficient, real density ) :
  MaterialModel(density, 1), youngModulus_(youngModulus), 
  poisson_(poissonCoefficient)
{
	// create material matrix
	real E  = youngModulus;
	real nu = poissonCoefficient;
	real rho = density;
  afm::RelativePointer ptr = afb::DenseMatrix::Create(6, 6);
	afb::DenseMatrix& D = absref<afb::DenseMatrix>(ptr);
	D.ClearAll();

	// matrix terms
	real c11 = E*(1-nu) / ((1-2*nu)*(1+nu));
	real c12 = E*nu / ((1-2*nu)*(1+nu));
	real G   = E / (2*(1+nu));

	S(0,0, c11)		S(0,1, c12)		S(0,2, c12)
	S(1,0, c12)		S(1,1, c11)		S(1,2, c12)
	S(2,0, c12)		S(2,1, c12)		S(2,2, c11)
													S(3,3, G)
																	S(4,4, G)
																					S(5,5, G)
	
	waveVelocity_ = sqrt(E * (1-nu) / (rho * (1+nu)*(1-2*nu)));
  materialMatrix_ = ptr;
}

adm::LinearElasticIsotropicModel::~LinearElasticIsotropicModel( void )
{
  afb::DenseMatrix& D = absref<afb::DenseMatrix>(materialMatrix_);
  D.Destroy();
  System::ModelMemory().Deallocate(materialMatrix_);
}

void adm::LinearElasticIsotropicModel::Destroy( void ) const
{
  const afb::DenseMatrix& D = absref<afb::DenseMatrix>(materialMatrix_);
  D.Destroy();
  System::ModelMemory().Deallocate(materialMatrix_);
}

adm::MaterialModel& adm::LinearElasticIsotropicModel::Clone( int ) const
{
  LinearElasticIsotropicModel *model = new LinearElasticIsotropicModel(
    youngModulus_, poisson_, Density());
	return *model;
}

const afb::DenseMatrix& 
  adm::LinearElasticIsotropicModel::GetMaterialTensor( void ) const
{
  return *(afb::DenseMatrix *)*materialMatrix_;
}

void adm::LinearElasticIsotropicModel::UpdateStresses( 
  adp::UpdatedPhysicalState& updatedState, 
  const adp::InfinitesimalState& currentState, const ada::AnalysisTimeline&, int)
{
	// sigma = D : epsilon
  auto& sigma = updatedState.Stress();
	auto& dSigma = updatedState.LastStressIncrement();
  const auto& D = absref<afb::DenseMatrix>(materialMatrix_);
  const auto& dEpsilon = currentState.LastStrainIncrement();
  afb::VectorProduct(dSigma, 1.0, D, dEpsilon);
  afb::VectorSum(sigma, sigma, +1.0, dSigma);
}

real adm::LinearElasticIsotropicModel::GetWavePropagationSpeed( void ) const
{
	return waveVelocity_;	
}

real adm::LinearElasticIsotropicModel::GetBulkModulus( void ) const
{
  return youngModulus_ / (3 * (1-2*poisson_));
}

real adm::LinearElasticIsotropicModel::GetShearModulus( void ) const
{
  return youngModulus_ / (2*(1+poisson_));
}

afu::Uuid adm::LinearElasticIsotropicModel::GetTypeId( void ) const
{
  // D6B50BF2-1FB3-4811-A8CF-12F6F6BC17E6
  int bytes[16] = {0xD6, 0xB5, 0x0B, 0xF2, 0x1F, 0xB3, 0x48, 0x11, 
                   0xA8, 0xCF, 0x12, 0xF6, 0xF6, 0xBC, 0x17, 0xE6};
  return afu::Uuid(bytes);
}

bool adm::LinearElasticIsotropicModel::IsGPUCapable( void ) const
{
  return true;
}

size_type adm::LinearElasticIsotropicModel::GetDataBlockSize( void ) const
{
  return sizeof(GPUMaterial);
}

void adm::LinearElasticIsotropicModel::InitializeGPUData( 
  void *baseDataAddress, real *density, real *waveSpeed, real *bulkModulus, 
  real *shearModulus, real *materialTensor )
{
  MaterialModel::InitializeGPUData(baseDataAddress, density, waveSpeed, 
    bulkModulus, shearModulus, materialTensor);
  GPUMaterial& mat = *(GPUMaterial *)baseDataAddress;
  mat.YoungModulus = youngModulus_;
  mat.PoissonCoefficient = poisson_;
  *waveSpeed = GetWavePropagationSpeed();
  *bulkModulus = GetBulkModulus();
  *shearModulus = GetShearModulus();

  // init material tensor
  real E  = youngModulus_;
  real nu = poisson_;
  real c11 = E*(1-nu) / ((1-2*nu)*(1+nu));
  real c12 = E*nu / ((1-2*nu)*(1+nu));
  real G  = E / (2*(1+nu));
  real *m = materialTensor;
  m[0]  = c11;	m[1]  = c12;  m[2]  = c12;  m[3]  = 0;  m[4]  = 0;  m[5]  = 0;
  m[6]  = c12;  m[7]  = c11;  m[8]  = c12;  m[9]  = 0;  m[10] = 0;  m[11] = 0;
  m[12] = c12;  m[13] = c12;  m[14] = c11;  m[15] = 0;  m[16] = 0;  m[17] = 0;
  m[18] = 0;    m[19] = 0;    m[20] = 0;    m[21] = G;  m[22] = 0;  m[23] = 0;
  m[24] = 0;    m[25] = 0;    m[26] = 0;    m[27] = 0;  m[28] = G;  m[29] = 0; 
  m[30] = 0;    m[31] = 0;    m[32] = 0;    m[33] = 0;  m[34] = 0;  m[35] = G;
}

adm::MaterialStrategy& adm::LinearElasticIsotropicModel::GetGPUStrategy( void )
{
  return LinearElasticIsotropicStrategy::GetInstance();
}
