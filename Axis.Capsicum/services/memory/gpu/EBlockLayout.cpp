#include "EBlockLayout.hpp"

namespace asmg = axis::services::memory::gpu;
namespace asmo = axis::services::memory;

#define ALLOCATION_ALIGNMENT  8ui64
#define ALIGN_BLOCK(x) \
  ((x + ALLOCATION_ALIGNMENT - 1) & ~(ALLOCATION_ALIGNMENT - 1))

#define BLOCK_OVERHEAD (sizeof(uint64) +  /* element id      */\
                        sizeof(int)    +  /* f-block size    */ \
                        sizeof(int))      /* total dof count */

asmg::EBlockLayout::EBlockLayout(size_type formulationBlockSize, 
  size_type materialBlockSize)
{
  formulationSize_ = ALIGN_BLOCK(formulationBlockSize);
  materialSize_    = ALIGN_BLOCK(materialBlockSize);
}

asmg::EBlockLayout::~EBlockLayout(void)
{
  // nothing to do here
}

asmo::MemoryLayout& asmg::EBlockLayout::Clone( void ) const
{
  return *new EBlockLayout(formulationSize_, materialSize_);
}

size_type asmg::EBlockLayout::DoGetSegmentSize( void ) const
{
  return formulationSize_ + materialSize_ + BLOCK_OVERHEAD;
}

void * asmg::EBlockLayout::GetFormulationBlockAddress( 
  void *elementBaseAddress ) const
{
  return (void *)((uint64)elementBaseAddress + BLOCK_OVERHEAD);
}

void * asmg::EBlockLayout::GetMaterialBlockAddress( 
  void *elementBaseAddress ) const
{
  return (void *)((uint64)elementBaseAddress + formulationSize_ + BLOCK_OVERHEAD);
}

uint64 asmg::EBlockLayout::GetElementId( void *elementBaseAddress ) const
{
  return *((uint64 *)((uint64)elementBaseAddress + sizeof(uint64)));
}

void asmg::EBlockLayout::InitMemoryBlock( void *elementBaseAddress, 
  uint64 elementId, int totalDofCount )
{
  uint64 *elementIdDescriptor = (uint64 *)elementBaseAddress;
  int *fBlockSizeDescriptor = (int *)&elementIdDescriptor[1];
  int *dofCountDescriptor = &fBlockSizeDescriptor[1];
  *elementIdDescriptor = elementId;
  *fBlockSizeDescriptor = formulationSize_;
  *dofCountDescriptor = totalDofCount;
}
