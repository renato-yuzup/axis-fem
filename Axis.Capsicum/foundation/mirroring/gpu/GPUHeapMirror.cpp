#include "GPUHeapMirror.hpp"
#include "foundation/mirroring/MemoryReflector.hpp"
#include "foundation/mirroring/HeapReflector.hpp"
#include "GPUMemoryAllocator.hpp"
#include "foundation/InvalidOperationException.hpp"
#include "foundation/memory/pointer.hpp"

namespace afmg = axis::foundation::mirroring::gpu;
namespace afc  = axis::foundation::computing;
namespace afm  = axis::foundation::memory;
namespace afmg = axis::foundation::mirroring::gpu;
namespace afmm = axis::foundation::mirroring;

afmg::GPUHeapMirror::GPUHeapMirror(afm::HeapBlockArena& memoryHeap, 
  afc::GPUDevice& targetDevice) : memoryHeap_(memoryHeap), 
  targetDevice_(targetDevice)
{
  reflector_ = nullptr;
  allocated_ = false;
}

afmg::GPUHeapMirror::~GPUHeapMirror(void)
{
  delete reflector_;
}

void afmg::GPUHeapMirror::Allocate( void )
{
  if (allocated_)
  {
    throw axis::foundation::InvalidOperationException();
  }
  afmg::GPUMemoryAllocator allocator(targetDevice_);
  afmm::HeapReflector::CloneStructure(allocator, memoryHeap_);
  reflector_ = &allocator.BuildReflector();
  afmm::HeapReflector::InitializeClone(*reflector_, memoryHeap_);
  allocated_ = true;
}

void afmg::GPUHeapMirror::Mirror( void )
{
  if (reflector_ == nullptr || !allocated_)
  {
    throw axis::foundation::InvalidOperationException();
  }
  afmm::HeapReflector::Mirror(*reflector_, memoryHeap_);
}

void afmg::GPUHeapMirror::Restore( void )
{
  if (reflector_ == nullptr || !allocated_)
  {
    throw axis::foundation::InvalidOperationException();
  }
  afmm::HeapReflector::Restore(memoryHeap_, *reflector_);
}

void afmg::GPUHeapMirror::Deallocate( void )
{
  if (reflector_ == nullptr || !allocated_)
  {
    throw axis::foundation::InvalidOperationException();
  }
  delete reflector_;
  reflector_ = nullptr;
  allocated_ = false;
}

void * afmg::GPUHeapMirror::GetHostBaseAddress( void ) const
{
  if (reflector_ == nullptr || !allocated_)
  {
    throw axis::foundation::InvalidOperationException();
  }
  return reflector_->GetBlockStartAddress(0);
  return &System::ModelMemory();
}

void * afmg::GPUHeapMirror::GetGPUBaseAddress( void ) const
{
  if (reflector_ == nullptr || !allocated_)
  {
    throw axis::foundation::InvalidOperationException();
  }
  return reflector_->GetBlockStartAddress(0);
}
