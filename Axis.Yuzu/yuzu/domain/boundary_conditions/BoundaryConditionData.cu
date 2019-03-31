#include "BoundaryConditionData.hpp"
#define MEMORY_WORD_LENGTH  16
#define ALIGN_BLOCK(x) \
  ((x + MEMORY_WORD_LENGTH - 1) & ~(MEMORY_WORD_LENGTH - 1))

namespace aydbc = axis::yuzu::domain::boundary_conditions;

GPU_ONLY aydbc::BoundaryConditionData::BoundaryConditionData(
  void *bcDataAddress, uint64 index, int bcDataSize)
{
  int baseSize = ALIGN_BLOCK(sizeof(real) + sizeof(uint64));
  int extendedSize = ALIGN_BLOCK(bcDataSize);
  int sz = baseSize + extendedSize;
  startingAddress_ = (void *)((uint64)bcDataAddress + index*sz);
}

GPU_ONLY aydbc::BoundaryConditionData::~BoundaryConditionData( void )
{
  // nothing to do here
}

GPU_ONLY real * aydbc::BoundaryConditionData::GetOutputBucket( void ) const
{
  return (real *)((uint64)startingAddress_ + sizeof(uint64));
}

GPU_ONLY uint64 aydbc::BoundaryConditionData::GetDofId( void ) const
{
  return *(uint64 *)startingAddress_;
}

GPU_ONLY void * aydbc::BoundaryConditionData::GetCustomData( void ) const
{
  return (void *)((uint64)startingAddress_ + ALIGN_BLOCK(sizeof(real) + sizeof(uint64)));
}
