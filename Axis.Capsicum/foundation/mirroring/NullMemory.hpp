#pragma once
#include "MemoryMirror.hpp"

namespace axis { namespace foundation { namespace mirroring {

/**
 * Defines a null space of memory, that is, no allocation or data transfer
 * occurs in host or external processing device.
 */
class NullMemory : public MemoryMirror
{
public:
  NullMemory(void);
  ~NullMemory(void);

  virtual void * GetHostBaseAddress( void ) const;
  virtual void * GetGPUBaseAddress( void ) const;
  virtual void Allocate( void );
  virtual void Deallocate( void );
  virtual void Mirror( void );
  virtual void Restore( void );
};

} } } // namespace axis::services::memory
