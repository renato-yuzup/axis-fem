// stdafx.h : include file for standard system include files,
// or project specific include files that are used frequently, but
// are changed infrequently
//

#pragma once

#include "targetver.h"

// Headers for CppUnitTest
#include "CppUnitTest.h"

#ifdef AXIS_DOUBLEPRECISION
#define REAL_TOLERANCE			1e-14
#else
#define REAL_TOLERANCE			2e-6f
#endif

// disable this unnecessary warning (floating-point truncation)
#pragma warning(disable: 4305)

using namespace Microsoft::VisualStudio::CppUnitTestFramework;
