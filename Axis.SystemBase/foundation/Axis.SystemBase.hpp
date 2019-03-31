#pragma once

// The following ifdef block is the standard way of creating macros which make exporting
// from a DLL simpler. All files within this DLL are compiled with the AXISSYSTEMBASE_EXPORTS
// symbol defined on the command line. This symbol should not be defined on any project
// that uses this DLL. This way any other project whose source files include this file see
// AXISSYSTEMBASE_API functions as being imported from a DLL, whereas this DLL sees symbols
// defined with this macro as being exported.
#ifdef AXISSYSTEMBASE_EXPORTS
#define AXISSYSTEMBASE_API __declspec(dllexport)
#else
#define AXISSYSTEMBASE_API __declspec(dllimport)
#endif

#if defined(__CUDA_ARCH__)
#define AXIS_GPU_LIBRARY extern
#else
#define AXIS_GPU_LIBRARY
#endif

#include "basic_types.hpp"
