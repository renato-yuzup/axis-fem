#include "InfinitesimalState.hpp"
#include "domain/elements/FiniteElement.hpp"
#include "foundation/blas/ColumnVector.hpp"
#include "foundation/blas/DenseMatrix.hpp"
#include "foundation/blas/blas.hpp"
#include "foundation/InvalidOperationException.hpp"
#include "System.hpp"
#include "foundation/memory/HeapBlockArena.hpp"
#include "foundation/memory/pointer.hpp"

namespace adp = axis::domain::physics;
namespace af = axis::foundation;
namespace afb = axis::foundation::blas;
namespace afm = axis::foundation::memory;

adp::InfinitesimalState::InfinitesimalState( void )
{
  _strain = NULLPTR;
  _stress = NULLPTR;
  _strainIncrement = NULLPTR;
  _stressIncrement = NULLPTR;
  _curDeformationGradient = NULLPTR;
  _lastDeformationGradient = NULLPTR;
}

adp::InfinitesimalState::~InfinitesimalState( void )
{  
  LastStrainIncrement().Destroy();
  LastStressIncrement().Destroy();
  Stress().Destroy();
  Strain().Destroy();
  DeformationGradient().Destroy();
  LastDeformationGradient().Destroy();
  System::ModelMemory().Deallocate(_strain);
  System::ModelMemory().Deallocate(_stress);
  System::ModelMemory().Deallocate(_strainIncrement);
  System::ModelMemory().Deallocate(_stressIncrement);
  System::ModelMemory().Deallocate(_curDeformationGradient);
  System::ModelMemory().Deallocate(_lastDeformationGradient);
  _strain = NULLPTR;
  _stress = NULLPTR;
  _strainIncrement = NULLPTR;
  _stressIncrement = NULLPTR;
  _curDeformationGradient = NULLPTR;
  _lastDeformationGradient = NULLPTR;
}

void adp::InfinitesimalState::Destroy( void ) const
{
  delete this;
}

void adp::InfinitesimalState::Reset( void )
{
  if (_strainIncrement == NULLPTR) _strainIncrement = afb::ColumnVector::Create(6);
  if (_stressIncrement == NULLPTR) _stressIncrement = afb::ColumnVector::Create(6);
  if (_strain == NULLPTR) _strain = afb::ColumnVector::Create(6);
  if (_stress == NULLPTR) _stress = afb::ColumnVector::Create(6);
  if (_curDeformationGradient == NULLPTR) _curDeformationGradient = afb::DenseMatrix::Create(3,3);
  if (_lastDeformationGradient == NULLPTR) _lastDeformationGradient = afb::DenseMatrix::Create(3,3);
  if (plasticStrain_ == NULLPTR) plasticStrain_ = afb::ColumnVector::Create(6);

  Strain().ClearAll();
  Stress().ClearAll();
  LastStrainIncrement().ClearAll();
  LastStressIncrement().ClearAll();
  PlasticStrain().ClearAll();
  effectivePlasticStrain_ = 0;

  afb::Identity(DeformationGradient());
  afb::Identity(LastDeformationGradient());
}

afb::DenseMatrix& adp::InfinitesimalState::DeformationGradient( void )
{
  if (_lastDeformationGradient == NULLPTR) 
  {
    throw af::InvalidOperationException(_T("Must reset properties first."));
  }
  return *(afb::DenseMatrix *)*_curDeformationGradient;
}

const afb::DenseMatrix& adp::InfinitesimalState::DeformationGradient( void ) const
{
  if (_curDeformationGradient == NULLPTR) throw af::InvalidOperationException(_T("Must reset properties first."));
  return *(afb::DenseMatrix *)*_curDeformationGradient;
}

const afb::DenseMatrix& adp::InfinitesimalState::LastDeformationGradient( void ) const
{
  if (_lastDeformationGradient == NULLPTR) throw af::InvalidOperationException(_T("Must reset properties first."));
  return *(afb::DenseMatrix *)*_lastDeformationGradient;
}

afb::DenseMatrix& adp::InfinitesimalState::LastDeformationGradient( void )
{
  if (_lastDeformationGradient == NULLPTR) throw af::InvalidOperationException(_T("Must reset properties first."));
  return *(afb::DenseMatrix *)*_lastDeformationGradient;
}

const afb::ColumnVector& adp::InfinitesimalState::Strain( void ) const
{
  if (_strain == NULLPTR) throw af::InvalidOperationException(_T("Must reset properties first."));
  return *(afb::ColumnVector *)*_strain;
}

afb::ColumnVector& adp::InfinitesimalState::Strain( void )
{
  if (_strain == NULLPTR) throw af::InvalidOperationException(_T("Must reset properties first."));
  return *(afb::ColumnVector *)*_strain;
}

const afb::ColumnVector& adp::InfinitesimalState::LastStrainIncrement( void ) const
{
  if (_strainIncrement == NULLPTR) throw af::InvalidOperationException(_T("Must reset properties first."));
  return *(afb::ColumnVector *)*_strainIncrement;
}
afb::ColumnVector& adp::InfinitesimalState::LastStrainIncrement( void )
{
  if (_strainIncrement == NULLPTR) throw af::InvalidOperationException(_T("Must reset properties first."));
  return *(afb::ColumnVector *)*_strainIncrement;
}

afb::ColumnVector& adp::InfinitesimalState::Stress( void )
{
  if (_stress == NULLPTR) throw af::InvalidOperationException(_T("Must reset properties first."));
  return *(afb::ColumnVector *)*_stress;
}
const afb::ColumnVector& adp::InfinitesimalState::Stress( void ) const
{
  if (_stress == NULLPTR) throw af::InvalidOperationException(_T("Must reset properties first."));
  return *(afb::ColumnVector *)*_stress;
}

afb::ColumnVector& adp::InfinitesimalState::LastStressIncrement( void )
{
  if (_stressIncrement == NULLPTR) throw af::InvalidOperationException(_T("Must reset properties first."));
  return *(afb::ColumnVector *)*_stressIncrement;
}
const afb::ColumnVector& adp::InfinitesimalState::LastStressIncrement( void ) const
{
  if (_stressIncrement == NULLPTR) throw af::InvalidOperationException(_T("Must reset properties first."));
  return *(afb::ColumnVector *)*_stressIncrement;
}

afb::ColumnVector&adp::InfinitesimalState::PlasticStrain( void )
{
  if (plasticStrain_ == NULLPTR) throw af::InvalidOperationException(_T("Must reset properties first."));
  return absref<afb::ColumnVector>(plasticStrain_);
}

const afb::ColumnVector& 
  adp::InfinitesimalState::PlasticStrain( void ) const
{
  if (plasticStrain_ == NULLPTR) throw af::InvalidOperationException(_T("Must reset properties first."));
  return absref<afb::ColumnVector>(plasticStrain_);
}

real&adp::InfinitesimalState::EffectivePlasticStrain( void )
{
  return effectivePlasticStrain_;
}

real adp::InfinitesimalState::EffectivePlasticStrain( void ) const
{
  return effectivePlasticStrain_;
}

void adp::InfinitesimalState::CopyFrom( const InfinitesimalState& source )
{
  if (_strainIncrement == NULLPTR) _strainIncrement = afb::ColumnVector::Create(6);
  if (_stressIncrement == NULLPTR) _stressIncrement = afb::ColumnVector::Create(6);
  if (_strain == NULLPTR) _strain = afb::ColumnVector::Create(6);
  if (_stress == NULLPTR) _stress = afb::ColumnVector::Create(6);
  if (_curDeformationGradient == NULLPTR) _curDeformationGradient = afb::DenseMatrix::Create(3,3);
  if (_lastDeformationGradient == NULLPTR) _lastDeformationGradient = afb::DenseMatrix::Create(3,3);
  auto& dEpsilon = absref<afb::ColumnVector>(_strainIncrement);
  auto& dSigma = absref<afb::ColumnVector>(_stressIncrement);
  auto& epsilon = absref<afb::ColumnVector>(_strain);
  auto& sigma = absref<afb::ColumnVector>(_stress);
  auto& newF = absref<afb::DenseMatrix>(_curDeformationGradient);
  auto& lastF = absref<afb::DenseMatrix>(_lastDeformationGradient);
  auto& ep = absref<afb::ColumnVector>(plasticStrain_);
  dEpsilon.CopyFrom(source.LastStrainIncrement());
  dSigma.CopyFrom(source.LastStressIncrement());
  epsilon.CopyFrom(source.Strain());
  sigma.CopyFrom(source.Stress());
  newF.CopyFrom(source.DeformationGradient());
  lastF.CopyFrom(source.LastDeformationGradient());
  ep.CopyFrom(source.PlasticStrain());
  effectivePlasticStrain_ = source.effectivePlasticStrain_;
}

void * adp::InfinitesimalState::operator new( size_t bytes )
{
  // It is supposed that the finite element object will remain in memory
  // until the end of the program. That's why we discard the relative
  // pointer. We ignore the fact that an exception might occur in
  // constructor because if it does happen, the program will end.
  afm::RelativePointer ptr = System::GlobalMemory().Allocate(bytes);
  return *ptr;
}

void * adp::InfinitesimalState::operator new( size_t, void *ptr )
{
  return ptr;
}

void adp::InfinitesimalState::operator delete( void * )
{
  // Since the relative pointer was discarded, we can't discard memory.
  // If it is really necessary, to free up resources, obliterating
  // memory pool is a better approach.
}

void adp::InfinitesimalState::operator delete( void *, void * )
{
  // Since the relative pointer was discarded, we can't discard memory.
  // If it is really necessary, to free up resources, obliterating
  // memory pool is a better approach.
}

afm::RelativePointer adp::InfinitesimalState::Create( void )
{
  afm::RelativePointer ptr = System::ModelMemory().Allocate(sizeof(InfinitesimalState));
  new (*ptr) InfinitesimalState();
  return ptr;
}
