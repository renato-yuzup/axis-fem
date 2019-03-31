#include "GPUHeapReflector.hpp"

namespace afc  = axis::foundation::computing;
namespace afmg = axis::foundation::mirroring::gpu;

afmg::GPUHeapReflector::GPUHeapReflector(afc::GPUDevice& device) : 
  device_(device)
{
  // nothing to do here
}

afmg::GPUHeapReflector::~GPUHeapReflector(void)
{
  int blockCount = GetBlockCount();
  for (int i = 0; i < blockCount; i++)
  {
    void *blockAddr = GetBlockStartAddress(i);
    device_.DeallocateMemory(blockAddr);
  }
}

void afmg::GPUHeapReflector::WriteToBlock( int destBlockIdx, 
  uint64 addressOffset, const void *sourceAddress, uint64 dataSize )
{
  device_.SetActive();
  void *devDestAddr = (void *)
    ((uint64)GetBlockStartAddress(destBlockIdx) + addressOffset);
  device_.SendToDevice(devDestAddr, sourceAddress, dataSize);
}

void afmg::GPUHeapReflector::Restore( void *destinationAddress, int srcBlockIdx, 
  uint64 addressOffset, uint64 dataSize )
{
  device_.SetActive();
  void *devSrcAddr = (void *)
    ((uint64)GetBlockStartAddress(srcBlockIdx) + addressOffset);
  device_.ReadFromDevice(destinationAddress, devSrcAddr, dataSize);
}
