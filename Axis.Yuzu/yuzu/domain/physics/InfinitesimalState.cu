#include "InfinitesimalState.hpp"
#include "yuzu/foundation/blas/linear_algebra.hpp"
#include "yuzu/foundation/memory/pointer.hpp"

namespace aydp = axis::yuzu::domain::physics;
namespace ayfb = axis::yuzu::foundation::blas;
namespace ayfm = axis::yuzu::foundation::memory;

GPU_ONLY aydp::InfinitesimalState::InfinitesimalState( void )
{
  // nothing to do here; private implementation
}

GPU_ONLY aydp::InfinitesimalState::~InfinitesimalState( void )
{  
  // nothing to do here
}

GPU_ONLY void aydp::InfinitesimalState::Reset( void )
{
	Strain().ClearAll();
	Stress().ClearAll();
	LastStrainIncrement().ClearAll();
	LastStressIncrement().ClearAll();
  PlasticStrain().ClearAll();
  ayfb::Identity(DeformationGradient());
  ayfb::Identity(LastDeformationGradient());
  effectivePlasticStrain_ = 0;
}

GPU_ONLY ayfb::DenseMatrix& aydp::InfinitesimalState::DeformationGradient( void )
{
  return yabsref<ayfb::DenseMatrix>(_curDeformationGradient);
}

GPU_ONLY const ayfb::DenseMatrix& aydp::InfinitesimalState::DeformationGradient( void ) const
{
  return yabsref<ayfb::DenseMatrix>(_curDeformationGradient);
}

GPU_ONLY const ayfb::DenseMatrix& aydp::InfinitesimalState::LastDeformationGradient( void ) const
{
  return yabsref<ayfb::DenseMatrix>(_lastDeformationGradient);
}

GPU_ONLY ayfb::DenseMatrix& aydp::InfinitesimalState::LastDeformationGradient( void )
{
  return yabsref<ayfb::DenseMatrix>(_lastDeformationGradient);
}

GPU_ONLY const ayfb::ColumnVector& aydp::InfinitesimalState::Strain( void ) const
{
  return yabsref<ayfb::ColumnVector>(_strain);
}

GPU_ONLY ayfb::ColumnVector& aydp::InfinitesimalState::Strain( void )
{
  return yabsref<ayfb::ColumnVector>(_strain);
}

GPU_ONLY const ayfb::ColumnVector& aydp::InfinitesimalState::LastStrainIncrement( void ) const
{
  return yabsref<ayfb::ColumnVector>(_strainIncrement);
}
GPU_ONLY ayfb::ColumnVector& aydp::InfinitesimalState::LastStrainIncrement( void )
{
  return yabsref<ayfb::ColumnVector>(_strainIncrement);
}

GPU_ONLY ayfb::ColumnVector& aydp::InfinitesimalState::Stress( void )
{
  return yabsref<ayfb::ColumnVector>(_stress);
}
GPU_ONLY const ayfb::ColumnVector& aydp::InfinitesimalState::Stress( void ) const
{
  return yabsref<ayfb::ColumnVector>(_stress);
}

GPU_ONLY ayfb::ColumnVector& aydp::InfinitesimalState::LastStressIncrement( void )
{
  return yabsref<ayfb::ColumnVector>(_stressIncrement);
}
GPU_ONLY const ayfb::ColumnVector& aydp::InfinitesimalState::LastStressIncrement( void ) const
{
  return yabsref<ayfb::ColumnVector>(_stressIncrement);
}

GPU_ONLY ayfb::ColumnVector& aydp::InfinitesimalState::PlasticStrain( void )
{
  return yabsref<ayfb::ColumnVector>(plasticStrain_);
}

GPU_ONLY const ayfb::ColumnVector& aydp::InfinitesimalState::PlasticStrain( void ) const
{
  return yabsref<ayfb::ColumnVector>(plasticStrain_);
}

GPU_ONLY real& aydp::InfinitesimalState::EffectivePlasticStrain( void )
{
  return effectivePlasticStrain_;
}

GPU_ONLY real aydp::InfinitesimalState::EffectivePlasticStrain( void ) const
{
  return effectivePlasticStrain_;
}

GPU_ONLY void aydp::InfinitesimalState::CopyFrom( const InfinitesimalState& source )
{
  Strain() = source.Strain();
  LastStrainIncrement() = source.LastStrainIncrement();
  Stress() = source.Stress();
  LastStressIncrement() = source.LastStressIncrement();
  PlasticStrain() = source.PlasticStrain();
  effectivePlasticStrain_ = source.EffectivePlasticStrain();
  DeformationGradient() = source.DeformationGradient();
  LastDeformationGradient() = source.LastDeformationGradient();
}
