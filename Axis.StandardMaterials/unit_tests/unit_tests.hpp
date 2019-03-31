#pragma once

// include unit test framework
#include <CppUnitTest.h>
#include <CppUnitTestAssert.h>
#include <CppUnitTestLogger.h>


using namespace Microsoft::VisualStudio::CppUnitTestFramework;


#ifdef AXIS_DOUBLEPRECISION
#define REAL_TOLERANCE			1e-14
#else
#define REAL_TOLERANCE			1e-6f
#endif

#define REAL_ROUNDOFF_TOLERANCE			(100 * REAL_TOLERANCE)
#define REAL_STRESS_TOLERANCE			(1E3 * REAL_ROUNDOFF_TOLERANCE)
