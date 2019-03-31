#pragma once
#include "System.hpp"
#include "foundation/memory/RelativePointer.hpp"
#include "foundation/memory/HeapBlockArena.hpp"

namespace axis {
#if !defined(AXIS_NO_MEMORY_ARENA)
  #define NULLPTR  axis::foundation::memory::RelativePointer::NullPtr

  template <class T>
  inline typename T *absptr(axis::foundation::memory::RelativePointer& ptr)
  {
    return (typename T *)*ptr;
  }
  template <class T>
  inline const typename T *absptr(const axis::foundation::memory::RelativePointer& ptr)
  {
    return (const typename T *)*ptr;
  }

  template <class T>
  inline typename T& absref(axis::foundation::memory::RelativePointer& ptr)
  {
    return *(typename T *)*ptr;
  }
  template <class T>
  inline const typename T& absref(const axis::foundation::memory::RelativePointer& ptr)
  {
    return *(const typename T *)*ptr;
  }
#else
  #define NULLPTR  NULL
  template <class T>
  inline typename T *absptr(T * ptr)
  {
    return ptr;
  }
  template <class T>
  inline const typename T *absptr(const T * ptr)
  {
    return ptr;
  }

  template <class T>
  inline typename T& absref(T * ptr)
  {
    return *ptr;
  }
  template <class T>
  inline const typename T& absref(const T * ptr)
  {
    return *ptr;
  }
#endif
}
