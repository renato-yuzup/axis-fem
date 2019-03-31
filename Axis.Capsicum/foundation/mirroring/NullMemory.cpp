#include "NullMemory.hpp"

namespace afmo = axis::foundation::mirroring;

afmo::NullMemory::NullMemory(void)
{
  // nothing to do here
}

afmo::NullMemory::~NullMemory(void)
{
  // nothing to do here
}

void * afmo::NullMemory::GetHostBaseAddress( void ) const
{
  return nullptr;
}

void * afmo::NullMemory::GetGPUBaseAddress( void ) const
{
  return nullptr;
}

void afmo::NullMemory::Allocate( void )
{
  // nothing to do here
}

void afmo::NullMemory::Deallocate( void )
{
  // nothing to do here
}

void afmo::NullMemory::Mirror( void )
{
  // nothing to do here
}

void afmo::NullMemory::Restore( void )
{
  // nothing to do here
}
