// Axis.YuzuTest.cpp : Defines the exported functions for the DLL application.
//

#include "stdafx.h"
#include "Axis.YuzuTest.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include "yuzu/foundation/memory/RelativePointer.hpp"

// This is an example of an exported variable
AXISYUZUTEST_API int nAxisYuzuTest=0;

// This is an example of an exported function.
AXISYUZUTEST_API int fnAxisYuzuTest(void)
{
	return 42;
}

__global__ void test(void)
{
  axis::yuzu::foundation::memory::RelativePointer ptr;

}

// This is the constructor of a class that has been exported.
// see Axis.YuzuTest.h for the class definition
CAxisYuzuTest::CAxisYuzuTest()
{
	return;
}
