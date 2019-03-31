#include "ReducedNumericalModel.hpp"
#include "yuzu/foundation/memory/pointer.hpp"

namespace ayda = axis::yuzu::domain::analyses;
namespace ayde = axis::yuzu::domain::elements;
namespace ayfm = axis::yuzu::foundation::memory;

GPU_ONLY ayda::ReducedNumericalModel::ReducedNumericalModel( void )
{
  // private implementation; nothing to do here
}

GPU_ONLY ayda::ReducedNumericalModel::~ReducedNumericalModel( void )
{
  // nothing to do here
}

GPU_ONLY const ayda::ModelDynamics& ayda::ReducedNumericalModel::Dynamics( void ) const
{
  return yabsref<ayda::ModelDynamics>(dynamics_);
}

GPU_ONLY ayda::ModelDynamics& ayda::ReducedNumericalModel::Dynamics( void )
{
  return yabsref<ayda::ModelDynamics>(dynamics_);
}

GPU_ONLY const ayda::ModelKinematics& ayda::ReducedNumericalModel::Kinematics( void ) const
{
  return yabsref<ayda::ModelKinematics>(kinematics_);
}

GPU_ONLY ayda::ModelKinematics& ayda::ReducedNumericalModel::Kinematics( void )
{
  return yabsref<ayda::ModelKinematics>(kinematics_);
}

GPU_ONLY size_type ayda::ReducedNumericalModel::GetElementCount( void ) const
{
  return elementCount_;
}

GPU_ONLY size_type ayda::ReducedNumericalModel::GetNodeCount( void ) const
{
  return nodeCount_;
}

GPU_ONLY const ayde::FiniteElement& ayda::ReducedNumericalModel::GetElement( size_type index ) const
{
  return yabsref<ayde::FiniteElement>(yabsptr<ayfm::RelativePointer>(elementArrayPtr_)[index]);
}

GPU_ONLY ayde::FiniteElement& ayda::ReducedNumericalModel::GetElement( size_type index )
{
  return yabsref<ayde::FiniteElement>(yabsptr<ayfm::RelativePointer>(elementArrayPtr_)[index]);
}

GPU_ONLY const ayfm::RelativePointer ayda::ReducedNumericalModel::GetElementPointer( size_type index ) const
{
  return yabsptr<ayfm::RelativePointer>(elementArrayPtr_)[index];
}

GPU_ONLY ayfm::RelativePointer ayda::ReducedNumericalModel::GetElementPointer( size_type index )
{
  return yabsptr<ayfm::RelativePointer>(elementArrayPtr_)[index];
}

GPU_ONLY const ayde::Node& ayda::ReducedNumericalModel::GetNode( size_type index ) const
{
  return yabsref<ayde::Node>(yabsptr<ayfm::RelativePointer>(nodeArrayPtr_)[index]);
}

GPU_ONLY ayde::Node& ayda::ReducedNumericalModel::GetNode( size_type index )
{
  return yabsref<ayde::Node>(yabsptr<ayfm::RelativePointer>(nodeArrayPtr_)[index]);
}

GPU_ONLY const ayfm::RelativePointer ayda::ReducedNumericalModel::GetNodePointer( size_type index ) const
{
  return yabsptr<ayfm::RelativePointer>(nodeArrayPtr_)[index];
}

GPU_ONLY ayfm::RelativePointer ayda::ReducedNumericalModel::GetNodePointer( size_type index )
{
  return yabsptr<ayfm::RelativePointer>(nodeArrayPtr_)[index];
}

GPU_ONLY const real * ayda::ReducedNumericalModel::GetElementOutputBucket( 
  size_type eIdx ) const
{
  const real *const *bucketArray = yabsptr<real *>(outputBucketArrayPtr_);
  return bucketArray[eIdx];
}

GPU_ONLY void ayda::ReducedNumericalModel::SetElementOutputBucket( 
  size_type eIdx, real *bucket )
{
  real **bucketArray = yabsptr<real *>(outputBucketArrayPtr_);
  bucketArray[eIdx] = bucket;
}
