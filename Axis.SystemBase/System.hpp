#pragma once
#include "Foundation/Axis.SystemBase.hpp"
#include "nocopy.hpp"
#include "foundation/fwd/memory_fwd.hpp"
#include "ApplicationEnvironment.hpp"

namespace axis {

namespace services { namespace management {
class SystemHook;
} } // namespace axis::services::management

class AXISSYSTEMBASE_API System
{
public:
  class AXISSYSTEMBASE_API SystemParameters
  {
  public:
    uint64 InitialStringMemorySize;
    uint64 InitialModelMemorySize;
    uint64 InitialGlobalMemorySize;
    uint64 WorkStackMemorySize;
    uint64 StringMemoryChunkSize;
    uint64 ModelMemoryChunkSize;
    uint64 GlobalMemoryChunkSize;
    int MaxThreadCount;

    SystemParameters(void);
    ~SystemParameters(void);
  };

  ~System(void);

  static void Initialize(void);
  static void Initialize(const SystemParameters& parameters);
  static void Finalize(void);

  static axis::foundation::memory::HeapStackArena& StringMemory(void);
  static axis::foundation::memory::FixedStackArena& StackMemory(void);
  static axis::foundation::memory::FixedStackArena& StackMemory(int index);
  static axis::foundation::memory::HeapBlockArena& ModelMemory(void);
  static axis::foundation::memory::HeapBlockArena& GlobalMemory(void);
  static axis::ApplicationEnvironment& Environment(void);
  static int GetMaxThreads(void);
  static void RegisterHook(int messageId, axis::services::management::SystemHook& hook);
  static void UnregisterHook(axis::services::management::SystemHook& hook);
  static void Broadcast(int messageId, void *messageDataPtr);
  static bool IsSystemReady();
private:
  struct Pimpl;

  System(void);
  DISALLOW_COPY_AND_ASSIGN(System);
  
  static Pimpl *pimpl_;
  static axis::foundation::memory::HeapStackArena **stringPool_;
  static axis::foundation::memory::FixedStackArena  **stackPool_;
  static axis::foundation::memory::HeapBlockArena *modelPool_;
  static axis::foundation::memory::HeapBlockArena *globalPool_;
  static axis::ApplicationEnvironment *env_;
  static int maxThreads_;
};

} // namespace axis

