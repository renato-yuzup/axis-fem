#include "stdafx.h"
#include "BiLinearPlasticityModel.hpp"
#include "foundation/memory/pointer.hpp"
#include "domain/analyses/AnalysisTimeline.hpp"
#include "domain/physics/InfinitesimalState.hpp"
#include "domain/physics/UpdatedPhysicalState.hpp"
#include "mechanics/continuum.hpp"
#include "foundation/blas/matrix_operations.hpp"
#include "foundation/blas/linear_algebra.hpp"
#include "BiLinearPlasticityStrategy.hpp"
#include "bilinear_plasticity_commands/BiLinearPlasticityGPUData.hpp"

namespace ada  = axis::domain::analyses;
namespace adm  = axis::domain::materials;
namespace admb = axis::domain::materials::bilinear_plasticity_commands;
namespace adp  = axis::domain::physics;
namespace am   = axis::mechanics;
namespace afb  = axis::foundation::blas;
namespace afu  = axis::foundation::uuids;

adm::BiLinearPlasticityModel::BiLinearPlasticityModel( real elasticModulus, 
  real poissonCoefficient, real yieldStress, real hardeningCoefficient, 
  real density, int numPoints ) : MaterialModel(density, numPoints)
{
  elasticModulus_ = elasticModulus;
  poisson_ = poissonCoefficient;
  yieldStress_ = yieldStress;
  hardeningCoeff_ = hardeningCoefficient;
  materialTensor_ = afb::DenseMatrix::Create(6, 6);
  
  // initialize material tensor
  auto& D = absref<afb::DenseMatrix>(materialTensor_);
  real E = elasticModulus, nu = poissonCoefficient;
  real c11 = E*(1-nu) / ((1-2*nu)*(1+nu));
  real c12 = E*nu / ((1-2*nu)*(1+nu));
  real G   = E / (2*(1+nu));
  D.ClearAll();
  D(0,0) = c11;		D(0,1) = c12;		D(0,2) = c12;
  D(1,0) = c12;		D(1,1) = c11;		D(1,2) = c12;
  D(2,0) = c12;		D(2,1) = c12;		D(2,2) = c11;
  D(3,3) = G;
  D(4,4) = G;
  D(5,5) = G;

  waveVelocity_ = sqrt(E * (1-nu) / (density * (1+nu)*(1-2*nu)));

  flowStressPtr_ = System::ModelMemory().Allocate(sizeof(real) * numPoints);
  flowStress_ = absptr<real>(flowStressPtr_);
  for (int i = 0; i < numPoints; i++)
  {
    flowStress_[i] = 0;
  }
}

adm::BiLinearPlasticityModel::~BiLinearPlasticityModel( void )
{
  auto& D = absref<afb::DenseMatrix>(materialTensor_);
  D.Destroy();
  System::ModelMemory().Deallocate(materialTensor_);
}

void adm::BiLinearPlasticityModel::Destroy( void ) const
{
  delete this;
}

const afb::DenseMatrix& 
  adm::BiLinearPlasticityModel::GetMaterialTensor( void ) const
{
  return absref<afb::DenseMatrix>(materialTensor_);
}

adm::MaterialModel& adm::BiLinearPlasticityModel::Clone( int numPoints ) const
{
  return *new BiLinearPlasticityModel(elasticModulus_, poisson_, 
    yieldStress_, hardeningCoeff_, Density(), numPoints);
}

void adm::BiLinearPlasticityModel::UpdateStresses( 
  adp::UpdatedPhysicalState& updatedState, 
  const adp::InfinitesimalState& currentState, 
  const ada::AnalysisTimeline& timeInfo, 
  int materialPointIndex )
{
  afb::DenseMatrix L(3, 3);             // velocity gradient
  afb::SymmetricMatrix D(3);            // deformation rate tensor
  afb::DenseMatrix W(3, 3);             // spin tensor
  auto& Fnext = currentState.DeformationGradient();
  auto& Fcur = currentState.LastDeformationGradient();
  real dt = timeInfo.NextTimeIncrement();

  // Calculate rate of deformation and spin tensors
  am::CalculateVelocityGradient(L, Fnext, Fcur, dt);
  afb::DecomposeSymmetric(D, L);
  afb::DecomposeSkew(W, L);

  // Do trial elastic step
  UpdateElasticStress(updatedState, currentState, D, W, dt);

  // Execute plastic correction, if needed
  ExecutePlasticCorrection(updatedState, currentState, D, W, dt, 
    materialPointIndex);

  // Increment stress
  auto& sigma = updatedState.Stress();
  auto& dSigma = updatedState.LastStressIncrement();
  sigma += dSigma;
}

real adm::BiLinearPlasticityModel::GetBulkModulus( void ) const
{
  return elasticModulus_ / (3*(1-2*poisson_));
}

real adm::BiLinearPlasticityModel::GetShearModulus( void ) const
{
  return elasticModulus_ / (2*(1+poisson_));
}

real adm::BiLinearPlasticityModel::GetWavePropagationSpeed( void ) const
{
  return waveVelocity_;
}

afu::Uuid adm::BiLinearPlasticityModel::GetTypeId( void ) const
{
  // 76CC253C-86D8-4C37-AD62-7F9E88FD8E3C
  const int uuid[] = {0x76, 0xCC, 0x25, 0x3C, 0x86, 0xD8, 0x4C, 0x37, 0xAD, 
    0x62, 0x7F, 0x9E, 0x88, 0xFD, 0x8E, 0x3C};
  return afu::Uuid(uuid);
}

void adm::BiLinearPlasticityModel::UpdateElasticStress( 
  adp::UpdatedPhysicalState& updatedState, 
  const adp::InfinitesimalState& currentState, 
  const afb::SymmetricMatrix& rateOfDeformationTensor,
  const afb::DenseMatrix& spinTensor,
  real timeIncrement )
{
  const auto& Ce = absref<afb::DenseMatrix>(materialTensor_);
  const auto& D = rateOfDeformationTensor;  // deformation rate tensor
  const auto& W = spinTensor;               // spin tensor
  afb::SymmetricMatrix stressRate(3);       // stress rate (wrt time)
  afb::SymmetricMatrix curSigma(3);         // total stress at current step

  // Write current stress in matrix form
  afb::TransformVoigtToSecondTensor(curSigma, currentState.Stress());

  // Calculate stress rate using objective rates
  CalculateStressRate(stressRate, Ce, curSigma, rateOfDeformationTensor, 
    spinTensor);

  // Calculate stress increment, but do not increment yet, as the increment
  // parcel will be corrected by the plasticity step
  auto& dSigma = updatedState.LastStressIncrement();
  afb::TransformSecondTensorToVoigt(dSigma, stressRate);
  dSigma.Scale(timeIncrement); // dSigma = stressRate * dt
}

void adm::BiLinearPlasticityModel::ExecutePlasticCorrection( 
  adp::UpdatedPhysicalState& updatedState, 
  const adp::InfinitesimalState& currentState, 
  axis::foundation::blas::SymmetricMatrix& rateOfElasticDeformationTensor,
  const axis::foundation::blas::DenseMatrix& spinTensor,
  real timeIncrement, int materialPointIndex )
{
  const auto& Ce = absref<afb::DenseMatrix>(materialTensor_);
  afb::ColumnVector trialStressVec(6);
  const auto& sigma0 = updatedState.Stress();  // current stress state (Voigt)
  auto& dSigma = updatedState.LastStressIncrement(); // (trial) stress increment
  afb::SymmetricMatrix trialStress(3);         // trial stress 2nd-order tensor
  afb::SymmetricMatrix flowDirection(3);       // flow direction unit tensor
  afb::ColumnVector flowDirectionVec(6);       // flow direction tensor (Voigt)
  real dLambda = 0;                            // plastic multiplier
  const real dt = timeIncrement;               

  // Calculate total trial deviatoric stress
  afb::VectorSum(trialStressVec, 1.0, sigma0, 1.0, dSigma);
  afb::TransformVoigtToSecondTensor(trialStress, trialStressVec);
  real pressure = trialStress.Trace() / 3.0;
  trialStress(0,0) -= pressure; 
  trialStress(1,1) -= pressure; 
  trialStress(2,2) -= pressure;

  // Determine yield function
  real equivStress = afb::DoubleContraction(1.0, trialStress, 1.0, trialStress);
  equivStress = sqrt(1.5 * equivStress);
  real equivPlasticStrain = currentState.EffectivePlasticStrain();
  const real& H = hardeningCoeff_;
  real yieldFun = equivStress - yieldStress_ - H*equivPlasticStrain;

  // Check consistency condition
  if (yieldFun > 0)   // plastic step occurring, correct stress 
  {                   // and update plastic strain state
    // Determine flow direction
    flowDirection = trialStress;
    flowDirection *= (3.0 / (2.0*equivStress));
    afb::TransformSecondTensorToVoigt(flowDirectionVec, flowDirection);
    flowDirectionVec(3) *= 2.0; 
    flowDirectionVec(4) *= 2.0; 
    flowDirectionVec(5) *= 2.0; 

    // Calculate plastic multiplier
    afb::ColumnVector aux(6), aux2(6);
    afb::TransformSecondTensorToVoigt(aux, rateOfElasticDeformationTensor);
    aux(3) *= 2.0; aux(4) *= 2.0; aux(5) *= 2.0; 
    afb::VectorProduct(aux2, 1.0, Ce, aux);
    real numerator = afb::VectorScalarProduct(flowDirectionVec, aux2);

    afb::VectorProduct(aux, 1.0, Ce, flowDirectionVec);
    real denominator = afb::VectorScalarProduct(flowDirectionVec, aux);
    denominator += H;
    dLambda = MATH_ABS(numerator / denominator);

    // Determine plastic strain increment (explicit integration)
    afb::ColumnVector dPlasticStrain(6);
    dPlasticStrain = flowDirectionVec;
    dPlasticStrain *= dLambda*dt;
    updatedState.PlasticStrain() += dPlasticStrain;

    // Update effective plastic strain (explicit integration)
    updatedState.EffectivePlasticStrain() += dLambda*dt;

    // Calculate rate of plastic deformation
    afb::SymmetricMatrix Dp(3);
    Dp = flowDirection; Dp *= dLambda;

    // Recalculate stress rate    
    auto& De = rateOfElasticDeformationTensor;
    afb::Sum(De, 1.0, De, -1.0, Dp);
    afb::SymmetricMatrix sigmaTensor(3);
    afb::SymmetricMatrix stressRate(3);
    afb::TransformVoigtToSecondTensor(sigmaTensor, sigma0);
    CalculateStressRate(stressRate, Ce, sigmaTensor, De, spinTensor);

    // Recalculate stress increment
    afb::TransformSecondTensorToVoigt(dSigma, stressRate);
    dSigma *= dt;
  }
}

void adm::BiLinearPlasticityModel::CalculateStressRate( 
  afb::SymmetricMatrix& stressRate, 
  const afb::DenseMatrix& elasticityMatrix,
  const afb::SymmetricMatrix& stressTensor,
  const afb::SymmetricMatrix& rateOfDeformationTensor, 
  const afb::DenseMatrix& spinTensor )
{
  auto& D = rateOfDeformationTensor;
  auto& W = spinTensor;
  afb::ColumnVector Dvec(6), jaumannVec(6);
  afb::TransformSecondTensorToVoigt(Dvec, D);
  Dvec(3) *= 2; Dvec(4) *= 2; Dvec(5) *= 2; 

  // Calculate Jaumann objective stress rate as a function of the rate of 
  // deformation tensor
  afb::VectorProduct(jaumannVec, 1.0, elasticityMatrix, Dvec);
  afb::TransformVoigtToSecondTensor(stressRate, jaumannVec);

  // Add to stress rate contribution regarding rigid body rotations
  afb::AccumulateProduct(stressRate, +1.0, W, stressTensor);
  afb::AccumulateProduct(stressRate, -1.0, stressTensor, W);
}

bool adm::BiLinearPlasticityModel::IsGPUCapable( void ) const
{
  return true;
}

size_type adm::BiLinearPlasticityModel::GetDataBlockSize( void ) const
{
  return sizeof(admb::BiLinearPlasticityGPUData);
}

void adm::BiLinearPlasticityModel::InitializeGPUData( void *baseDataAddress, 
  real *density, real *waveSpeed, real *bulkModulus, real *shearModulus, 
  real *materialTensor )
{
  MaterialModel::InitializeGPUData(baseDataAddress, density, waveSpeed, 
    bulkModulus, shearModulus, materialTensor);
  admb::BiLinearPlasticityGPUData& matData = 
    *(admb::BiLinearPlasticityGPUData *)baseDataAddress;
  matData.YieldStress = yieldStress_;
  matData.HardeningCoefficient = hardeningCoeff_;
  *waveSpeed = GetWavePropagationSpeed();
  *bulkModulus = GetBulkModulus();
  *shearModulus = GetShearModulus();

  // init material tensor
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

adm::MaterialStrategy& adm::BiLinearPlasticityModel::GetGPUStrategy( void )
{
  return BiLinearPlasticityStrategy::GetInstance();
}
