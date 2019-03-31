#include "ElementInitializerCommand.hpp"
#include "services/memory/MemoryGrid.hpp"
#include "foundation/ArgumentException.hpp"
#include "foundation/InvalidOperationException.hpp"

namespace asmc = axis::services::memory::commands;
namespace asmo = axis::services::memory;
namespace afm  = axis::foundation::memory;

asmc::ElementInitializerCommand::ElementInitializerCommand( 
  void **associatedDataPtr, uint64 elementCount )
{
  elementCount_ = elementCount;
  associatedDataPtr_ = new void *[elementCount];
  for (uint64 i = 0; i < elementCount; i++)
  {
    associatedDataPtr_[i] = associatedDataPtr[i];
  }
  blockLayout_ = nullptr;
}

asmc::ElementInitializerCommand::~ElementInitializerCommand(void)
{
  delete [] associatedDataPtr_;
  delete blockLayout_;
  associatedDataPtr_ = nullptr;
  blockLayout_ = nullptr;
}

asmc::MemoryCommand& asmc::ElementInitializerCommand::Clone( void ) const
{
  return DoSlice(associatedDataPtr_, elementCount_);
}

asmc::ElementInitializerCommand& asmc::ElementInitializerCommand::Slice( 
  uint64 offset, uint64 elementCount ) const
{
  if (offset + elementCount > elementCount_)
  {
    throw axis::foundation::ArgumentException();
  }
  return DoSlice(&associatedDataPtr_[offset], elementCount);
}

uint64 asmc::ElementInitializerCommand::GetElementCount( void ) const
{
  return elementCount_;
}

void asmc::ElementInitializerCommand::Execute( void *cpuBaseAddress, 
  void *gpuBaseAddress )
{
  if (blockLayout_ == nullptr)
  {
    throw axis::foundation::InvalidOperationException();
  }
  asmo::MemoryGrid grid(cpuBaseAddress, gpuBaseAddress, *blockLayout_, gridSize_);
  for (uint64 i = 0; i < elementCount_; i++)
  {
    void *targetAddress = grid.GetHostCellAddress(i);
    void *mirroredAddress = grid.GetMirroredCellAddress(i);
    DoExecute(targetAddress, mirroredAddress, associatedDataPtr_[i]);
  }
}

void asmc::ElementInitializerCommand::SetTargetGridLayout( 
  const asmo::MemoryLayout& blockLayout, uint64 gridSize )
{
  blockLayout_ = &blockLayout.Clone();
  gridSize_ = gridSize;
}
