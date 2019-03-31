#include "stdafx.h"
#include "MyElementsHook.hpp"
#include "yuzu/foundation/memory/RelativePointer.hpp"

namespace asmg = axis::services::management;
namespace ayfm = axis::yuzu::foundation::memory;

asmg::MyElementsHook::MyElementsHook(void)
{
  // nothing to do here
}

asmg::MyElementsHook::~MyElementsHook(void)
{
  // nothing to do here
}

void asmg::MyElementsHook::ProcessMessage( int, void *dataPtr )
{
  ayfm::SetGPUArena(dataPtr);
}
