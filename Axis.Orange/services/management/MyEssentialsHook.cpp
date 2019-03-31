#include "stdafx.h"
#include "MyEssentialsHook.hpp"
#include "yuzu/foundation/memory/RelativePointer.hpp"

namespace asmg = axis::services::management;
namespace ayfm = axis::yuzu::foundation::memory;

asmg::MyEssentialsHook::MyEssentialsHook(void)
{
  // nothing to do here
}

asmg::MyEssentialsHook::~MyEssentialsHook(void)
{
  // nothing to do here
}

void asmg::MyEssentialsHook::ProcessMessage( int, void *dataPtr )
{
  ayfm::SetGPUArena(dataPtr);
}
