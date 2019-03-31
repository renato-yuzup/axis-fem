#include "ElementData.hpp"

namespace ayde = axis::yuzu::domain::elements;

GPU_ONLY ayde::ElementData::ElementData(void *segmentAddress, uint64 blockIndex,
                                       uint64 elementBlockSize)
{
  elementBlockAddr_ = (void *)
    ((uint64)segmentAddress + blockIndex*elementBlockSize);
  elementId_ = *(uint64 *)elementBlockAddr_;
  formulationBlockSize_ = *(int *)((uint64)elementBlockAddr_ + sizeof(uint64));
  dofCount_ = *(int *)((uint64)elementBlockAddr_ + sizeof(uint64) + sizeof(int));
  fBlockAddr_ = (void *)
    ((uint64)elementBlockAddr_ + sizeof(uint64) + 2*sizeof(int));
  mBlockAddr_ = (void *)((uint64)fBlockAddr_ + formulationBlockSize_);
}

GPU_ONLY ayde::ElementData::~ElementData( void )
{
  // nothing to do here
}

GPU_ONLY uint64 ayde::ElementData::GetId( void ) const
{
  return elementId_;
}

GPU_ONLY real& ayde::ElementData::ArtificialEnergy( void )
{
  return *(real *)((uint64)fBlockAddr_ + dofCount_*dofCount_*sizeof(real));
}

GPU_ONLY const real& ayde::ElementData::ArtificialEnergy( void ) const
{
  return *(const real *)
    ((uint64)fBlockAddr_ + dofCount_*dofCount_*sizeof(real));
}

GPU_ONLY real * ayde::ElementData::GetOutputBuffer( void )
{
  return (real *)fBlockAddr_;
}

GPU_ONLY const real * ayde::ElementData::GetOutputBuffer( void ) const
{
  return (const real *)fBlockAddr_;
}

GPU_ONLY void * ayde::ElementData::GetFormulationBlock( void )
{
  return (void *)
    ((uint64)fBlockAddr_ + (dofCount_*dofCount_ + 1)*sizeof(real));
}

GPU_ONLY const void * ayde::ElementData::GetFormulationBlock( void ) const
{
  return (const void *)
    ((uint64)fBlockAddr_ + (dofCount_*dofCount_ + 1)*sizeof(real));
}

GPU_ONLY real ayde::ElementData::Density( void ) const
{
  return *(real *)mBlockAddr_;
}

GPU_ONLY real& ayde::ElementData::WaveSpeed( void )
{
  return *(real *)((uint64)mBlockAddr_ + sizeof(real));
}

GPU_ONLY const real& ayde::ElementData::WaveSpeed( void ) const
{
  return *(const real *)((uint64)mBlockAddr_ + sizeof(real));
}

GPU_ONLY real& ayde::ElementData::BulkModulus( void )
{
  return *(real *)((uint64)mBlockAddr_ + 2*sizeof(real));
}

GPU_ONLY const real& ayde::ElementData::BulkModulus( void ) const
{
  return *(const real *)((uint64)mBlockAddr_ + 2*sizeof(real));
}

GPU_ONLY real& ayde::ElementData::ShearModulus( void )
{
  return *(real *)((uint64)mBlockAddr_ + 3*sizeof(real));
}

GPU_ONLY const real& ayde::ElementData::ShearModulus( void ) const
{
  return *(const real *)((uint64)mBlockAddr_ + 3*sizeof(real));
}

GPU_ONLY real * ayde::ElementData::MaterialTensor( void )
{
  return (real *)((uint64)mBlockAddr_ + 4*sizeof(real));
}

GPU_ONLY const real *ayde::ElementData::MaterialTensor( void ) const
{
  return (const real *)((uint64)mBlockAddr_ + 4*sizeof(real));
}

GPU_ONLY void * ayde::ElementData::GetMaterialBlock( void )
{
  return (void *)((uint64)mBlockAddr_ + 40*sizeof(real));
}

GPU_ONLY const void *ayde::ElementData::GetMaterialBlock( void ) const
{
  return (const void *)((uint64)mBlockAddr_ + 40*sizeof(real));
}
