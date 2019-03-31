#include "stdafx.h"
#include "MyMaterialsHook.hpp"
#include "yuzu/foundation/memory/RelativePointer.hpp"

namespace asmg = axis::services::management;
namespace ayfm = axis::yuzu::foundation::memory;

asmg::MyMaterialsHook::MyMaterialsHook(void)
{
  // nothing to do here
}

asmg::MyMaterialsHook::~MyMaterialsHook(void)
{
  // nothing to do here
}

void asmg::MyMaterialsHook::ProcessMessage( int, void *dataPtr )
{
  ayfm::SetGPUArena(dataPtr);
}
