#pragma once

// include unit test framework
#include <CppUnitTest.h>
#include <CppUnitTestAssert.h>
#include <CppUnitTestLogger.h>


using namespace Microsoft::VisualStudio::CppUnitTestFramework;


#ifdef AXIS_DOUBLEPRECISION
#define REAL_TOLERANCE			1e-13
#else
#define REAL_TOLERANCE			1e-5f
#endif

#define REAL_ROUNDOFF_TOLERANCE			(100 * REAL_TOLERANCE)
