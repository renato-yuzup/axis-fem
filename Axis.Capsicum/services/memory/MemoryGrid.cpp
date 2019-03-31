#include "MemoryGrid.hpp"
#include "foundation/OutOfBoundsException.hpp"
#include "MemoryLayout.hpp"

namespace asmo = axis::services::memory;

asmo::MemoryGrid::MemoryGrid( void *hostBlockArrayAddress, 
  void *mirroredBlockArrayAddress, const MemoryLayout& layout, 
  uint64 cellCount ) : layout_(layout.Clone())
{
  hostBlockArrayAddress_ = hostBlockArrayAddress;
  mirroredBlockArrayAddress_ = mirroredBlockArrayAddress;
  cellCount_ = cellCount;
}

asmo::MemoryGrid::~MemoryGrid(void)
{
  delete &layout_;
}

void * asmo::MemoryGrid::GetHostCellAddress( uint64 index )
{
  if (index >= cellCount_)
  {
    throw axis::foundation::OutOfBoundsException();
  }
  return (void *)
    ((uint64)hostBlockArrayAddress_ + index*layout_.GetSegmentSize());
}

const void * asmo::MemoryGrid::GetHostCellAddress( uint64 index ) const
{
  if (index >= cellCount_)
  {
    throw axis::foundation::OutOfBoundsException();
  }
  return (void *)
    ((uint64)hostBlockArrayAddress_ + index*layout_.GetSegmentSize());
}

void * asmo::MemoryGrid::GetMirroredCellAddress( uint64 index )
{
  if (index >= cellCount_)
  {
    throw axis::foundation::OutOfBoundsException();
  }
  return (void *)
    ((uint64)mirroredBlockArrayAddress_ + index*layout_.GetSegmentSize());
}

const void * asmo::MemoryGrid::GetMirroredCellAddress( uint64 index ) const
{
  if (index >= cellCount_)
  {
    throw axis::foundation::OutOfBoundsException();
  }
  return (void *)
    ((uint64)mirroredBlockArrayAddress_ + index*layout_.GetSegmentSize());
}

uint64 asmo::MemoryGrid::GetCellCount( void ) const
{
  return cellCount_;
}
