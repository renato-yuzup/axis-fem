#include "stdafx.h"
#include "MySolverHook.hpp"
#include "yuzu/foundation/memory/RelativePointer.hpp"

namespace asmg = axis::services::management;
namespace ayfm = axis::yuzu::foundation::memory;

asmg::MySolverHook::MySolverHook(void)
{
  // nothing to do here
}

asmg::MySolverHook::~MySolverHook(void)
{
  // nothing to do here
}

void asmg::MySolverHook::ProcessMessage( int, void *dataPtr )
{
  ayfm::SetGPUArena(dataPtr);
}
