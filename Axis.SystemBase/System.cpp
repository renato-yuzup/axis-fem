#include "System.hpp"
#include <mutex>
#include <omp.h>
#include <map>
#include "services/management/SystemHook.hpp"
#include "foundation/memory/FixedStackArena.hpp"
#include "foundation/memory/HeapBlockArena.hpp"
#include "foundation/memory/HeapStackArena.hpp"
#include "foundation/InvalidOperationException.hpp"
#include "foundation/OutOfBoundsException.hpp"
#include "foundation/ArgumentException.hpp"
#include "system_messages.hpp"

#define DEFAULT_STRING_MEMSIZE      5000000
#define DEFAULT_STRING_CHUNKSIZE    2000000
#define DEFAULT_STACK_MEMSIZE      10000000
#define DEFAULT_MODEL_MEMSIZE      80000000
#define DEFAULT_MODEL_CHUNKSIZE    60000000
#define DEFAULT_GLOBAL_MEMSIZE     40000000
#define DEFAULT_GLOBAL_CHUNKSIZE   40000000

#define STRING_ARENA_POOL_INDEX     0
#define MODEL_ARENA_POOL_INDEX      1
#define GLOBAL_ARENA_POOL_INDEX     2

namespace asmg = axis::services::management;
namespace afm = axis::foundation::memory;

struct axis::System::Pimpl
{
public:
  std::mutex mutex_;
  typedef std::map<asmg::SystemHook *, int> hook_set;
  hook_set hooks_;
};

/************************************************************************/
/***********          BEGIN SystemParameters                  ***********/
/************************************************************************/
axis::System::SystemParameters::SystemParameters( void )
{
  // init with default parameters
  InitialStringMemorySize = DEFAULT_STRING_MEMSIZE;
  InitialModelMemorySize = DEFAULT_MODEL_MEMSIZE;
  InitialGlobalMemorySize = DEFAULT_GLOBAL_MEMSIZE;
  StringMemoryChunkSize = DEFAULT_STRING_CHUNKSIZE;
  ModelMemoryChunkSize = DEFAULT_MODEL_CHUNKSIZE;
  GlobalMemoryChunkSize = DEFAULT_GLOBAL_CHUNKSIZE;
  WorkStackMemorySize = DEFAULT_STACK_MEMSIZE;
  MaxThreadCount = omp_get_max_threads();
}
axis::System::SystemParameters::~SystemParameters( void )
{
  // nothing to do here
}
/************************************************************************/
/***********           END SystemParameters                   ***********/
/************************************************************************/

// Static members initialization
afm::HeapStackArena ** axis::System::stringPool_ = nullptr;
afm::FixedStackArena ** axis::System::stackPool_  = nullptr;
afm::HeapBlockArena * axis::System::modelPool_  = nullptr;
afm::HeapBlockArena * axis::System::globalPool_ = nullptr;
axis::ApplicationEnvironment * axis::System::env_ = nullptr;
int axis::System::maxThreads_ = 0;
axis::System::Pimpl * axis::System::pimpl_ = new axis::System::Pimpl();

axis::System::System( void )
{
  // nothing to do here
}

axis::System::~System( void )
{
  // nothing to do here
}

void axis::System::Initialize( void )
{
  Initialize(SystemParameters());
}

bool axis::System::IsSystemReady()
{
  return (stringPool_ != nullptr);
}

void axis::System::Initialize( const SystemParameters& parameters )
{
  std::lock_guard<std::mutex> guard(pimpl_->mutex_);
  if (stringPool_ != nullptr) 
  {
    throw axis::foundation::InvalidOperationException();
  }

  stringPool_ = new afm::HeapStackArena *[parameters.MaxThreadCount];
  for (int i = 0; i < parameters.MaxThreadCount; i++)
  {
    stringPool_[i] = &afm::HeapStackArena::Create(parameters.InitialStringMemorySize, 
      parameters.StringMemoryChunkSize);
  }

  stackPool_ = new afm::FixedStackArena *[parameters.MaxThreadCount];
  for (int i = 0; i < parameters.MaxThreadCount; i++)
  {
    stackPool_[i] = &afm::FixedStackArena::Create(parameters.WorkStackMemorySize);
  }

  modelPool_ = &afm::HeapBlockArena::Create(parameters.InitialModelMemorySize, 
                                            parameters.ModelMemoryChunkSize, 
                                            MODEL_ARENA_POOL_INDEX);
  globalPool_ = &afm::HeapBlockArena::Create(parameters.InitialGlobalMemorySize, 
                                             parameters.GlobalMemoryChunkSize, 
                                             GLOBAL_ARENA_POOL_INDEX);
  env_ = new axis::ApplicationEnvironment();
  env_->RefreshFromSystem();
  maxThreads_ = parameters.MaxThreadCount;
}

void axis::System::Finalize( void )
{
  std::lock_guard<std::mutex> guard(pimpl_->mutex_);
  if (stringPool_ == nullptr) 
  {
    throw axis::foundation::InvalidOperationException();
  }
  for (int i = 0; i < maxThreads_; i++)
  {
    stringPool_[i]->Destroy();
  }
  for (int i = 0; i < maxThreads_; i++)
  {
    stackPool_[i]->Destroy();
  }
  delete [] stackPool_;
  delete [] stringPool_;
  modelPool_->Destroy();
  globalPool_->Destroy();
  delete env_;

  stringPool_ = nullptr;
  stackPool_ = nullptr;
  modelPool_ = nullptr;
  globalPool_ = nullptr;
  env_ = nullptr;
}

afm::HeapStackArena& axis::System::StringMemory( void )
{
  int threadIndex = omp_get_thread_num();
  return *stringPool_[threadIndex];
}

afm::FixedStackArena& axis::System::StackMemory( void )
{
  int threadIndex = omp_get_thread_num();
  return *stackPool_[threadIndex];
}

afm::FixedStackArena& axis::System::StackMemory( int index )
{
  if (index < 0 || index >= maxThreads_)
  {
    throw axis::foundation::OutOfBoundsException();
  }
  return *stackPool_[index];
}

afm::HeapBlockArena& axis::System::ModelMemory( void )
{
  return *modelPool_;
}

afm::HeapBlockArena& axis::System::GlobalMemory( void )
{
  return *globalPool_;
}

int axis::System::GetMaxThreads( void )
{
  return maxThreads_ > omp_get_max_threads()? maxThreads_ : omp_get_max_threads();
}

axis::ApplicationEnvironment& axis::System::Environment( void )
{
  return *env_;
}

void axis::System::RegisterHook( int messageId, asmg::SystemHook& hook )
{
  if (pimpl_->hooks_.find(&hook) != pimpl_->hooks_.end())
  { // hook already registered
    throw axis::foundation::ArgumentException();
  }
  pimpl_->hooks_[&hook] = messageId;
}

void axis::System::UnregisterHook( asmg::SystemHook& hook )
{
  if (pimpl_->hooks_.find(&hook) == pimpl_->hooks_.end())
  { // hook not found
    throw axis::foundation::ArgumentException();
  }
  pimpl_->hooks_.erase(&hook);
}

void axis::System::Broadcast( int messageId, void *messageDataPtr )
{
  Pimpl::hook_set::iterator end = pimpl_->hooks_.end();
  Pimpl::hook_set::iterator it = pimpl_->hooks_.begin();
  for (; it != end; ++it)
  {
    asmg::SystemHook &hook = *it->first;
    int desiredMsgId = it->second;
    if (desiredMsgId == messageId || desiredMsgId == AXIS_SYS_ALL_MESSAGES)
    {
      hook.ProcessMessage(messageId, messageDataPtr);
    }
  }
}
