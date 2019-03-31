#include "NeoHookeanModel.hpp"
#include "domain/materials/neohookean_commands/NeoHookeanGPUData.hpp"
#include "domain/physics/InfinitesimalState.hpp"
#include "domain/physics/UpdatedPhysicalState.hpp"
#include "foundation/memory/pointer.hpp"
#include "foundation/blas/linear_algebra.hpp"
#include "NeoHookeanStrategy.hpp"

namespace adm  = axis::domain::materials;
namespace admn = axis::domain::materials::neohookean_commands;
namespace adp  = axis::domain::physics;
namespace afb  = axis::foundation::blas;
namespace afm  = axis::foundation::memory;
namespace afu  = axis::foundation::uuids;

adm::NeoHookeanModel::NeoHookeanModel(real youngModulus, real poissonCoefficient, 
  real density) : MaterialModel(density, 1)
{
  elasticModulus_ = youngModulus;
  poisson_ = poissonCoefficient;
  const real nu = poissonCoefficient;
  const real E = youngModulus;
  // calculate Lamé constants
  lambdaLameCoeff_ = nu*E / ((1+nu)*(1-2*nu));
  muLameCoeff_ = E / (2*(1+nu));

  materialMatrix_ = afb::DenseMatrix::Create(6, 6);
  waveVelocity_ = 2*muLameCoeff_ * (3*lambdaLameCoeff_ + muLameCoeff_) / 
    (density * (3*lambdaLameCoeff_ + 2*muLameCoeff_));
}

adm::NeoHookeanModel::~NeoHookeanModel( void )
{
  absref<afb::DenseMatrix>(materialMatrix_).Destroy();
  System::ModelMemory().Deallocate(materialMatrix_);
}

void adm::NeoHookeanModel::Destroy( void ) const
{
  delete this;
}

adm::MaterialModel& adm::NeoHookeanModel::Clone( int ) const
{
  return *new NeoHookeanModel(elasticModulus_, poisson_, Density());
}

const afb::DenseMatrix& adm::NeoHookeanModel::GetMaterialTensor( void ) const
{
  return absref<afb::DenseMatrix>(materialMatrix_);
}

void adm::NeoHookeanModel::UpdateStresses( 
  adp::UpdatedPhysicalState& updatedState, 
  const adp::InfinitesimalState& currentState, 
  const axis::domain::analyses::AnalysisTimeline& timeInfo, int )
{
  // calculate coefficients
  auto& F = currentState.DeformationGradient();  
  const real J = afb::Determinant3D(F);
  const real c1 = muLameCoeff_ / J;
  const real c2 = (lambdaLameCoeff_*log(J) - muLameCoeff_) / J;

  // Calculate left Cauchy-Green strain tensor (B -- symmetric)
  afb::SymmetricMatrix B(3,3);
  afb::Product(B, 1.0, F, afb::NotTransposed, F, afb::Transposed);
  
  // Calculate Cauchy stress tensor
  afb::SymmetricMatrix sigma(3,3);
  sigma.CopyFrom(B);
  sigma.Scale(c1);
  sigma(0,0) += c2; sigma(1,1) += c2; sigma(2,2) += c2;

  const real sxx = sigma(0,0);
  const real syy = sigma(1,1);
  const real szz = sigma(2,2);
  const real syz = sigma(1,2);
  const real sxz = sigma(0,2);
  const real sxy = sigma(0,1);

  // Update stress in material point
  auto& updatedStress = updatedState.Stress();
  updatedStress(0) = sxx;
  updatedStress(1) = syy;
  updatedStress(2) = szz;
  updatedStress(3) = syz;
  updatedStress(4) = sxz;
  updatedStress(5) = sxy;
}

real adm::NeoHookeanModel::GetBulkModulus( void ) const
{
  return lambdaLameCoeff_ + 2.0*muLameCoeff_ / 3.0;
}

real adm::NeoHookeanModel::GetShearModulus( void ) const
{
  return muLameCoeff_;
}

real adm::NeoHookeanModel::GetWavePropagationSpeed( void ) const
{
  return waveVelocity_;
}

afu::Uuid adm::NeoHookeanModel::GetTypeId( void ) const
{
  // E72880F9-4F10-47FD-AD03-D8B0F71D4910
  int uuid[] = {0xE7,0x28,0x80,0xF9,0x4F,0x10,0x47,0xFD,0xAD,0x03,0xD8,0xB0,
                0xF7,0x1D,0x49,0x10};
  return afu::Uuid(uuid);
}

bool adm::NeoHookeanModel::IsGPUCapable( void ) const
{
  return true;
}

size_type adm::NeoHookeanModel::GetDataBlockSize( void ) const
{
  return sizeof(admn::NeoHookeanGPUData);
}

void adm::NeoHookeanModel::InitializeGPUData( void *baseDataAddress, 
  real *density, real *waveSpeed, real *bulkModulus, real *shearModulus, 
  real *materialTensor )
{
  MaterialModel::InitializeGPUData(baseDataAddress, density, waveSpeed, 
    bulkModulus, shearModulus, materialTensor);
  admn::NeoHookeanGPUData& matData = *(admn::NeoHookeanGPUData *)baseDataAddress;
  matData.LambdaCoefficient = lambdaLameCoeff_;
  matData.MuCoefficient = muLameCoeff_;
  *waveSpeed = GetWavePropagationSpeed();
  *bulkModulus = GetBulkModulus();
  *shearModulus = GetShearModulus();

  // This material tensor is not accurate; however, it (might) will be only 
  // used by hourglass routines 
  real E  = elasticModulus_;
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

adm::MaterialStrategy& adm::NeoHookeanModel::GetGPUStrategy( void )
{
  return NeoHookeanStrategy::GetInstance();
}
