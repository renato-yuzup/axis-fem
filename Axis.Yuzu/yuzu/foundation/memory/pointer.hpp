#pragma once
#include "yuzu/common/gpu.hpp"
#include "yuzu/foundation/memory/RelativePointer.hpp"
#include "yuzu/foundation/memory/HeapBlockArena.hpp"

namespace axis {

template <class T>
inline GPU_ONLY typename T *yabsptr(axis::yuzu::foundation::memory::RelativePointer& ptr)
{
  return (typename T *)*ptr;
}
template <class T>
inline GPU_ONLY const typename T *yabsptr(const axis::yuzu::foundation::memory::RelativePointer& ptr)
{
  return (const typename T *)*ptr;
}

template <class T>
inline GPU_ONLY typename T& yabsref(axis::yuzu::foundation::memory::RelativePointer& ptr)
{
  return *(typename T *)*ptr;
}
template <class T>
inline GPU_ONLY const typename T& yabsref(const axis::yuzu::foundation::memory::RelativePointer& ptr)
{
  return *(const typename T *)*ptr;
}

} // namespace axis
