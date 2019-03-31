#pragma once

#include <cuda.h>
#include <cuda_runtime.h>

#if defined(__CUDACC__) || defined(__CUDA_ARCH__)
#define CPU_ONLY  __host__
#define GPU_ONLY  __device__
#define GPU_READY __host__ __device__
#else
#define CPU_ONLY
#define GPU_ONLY
#define GPU_READY
#endif

#define AXIS_YUZU_MAX_THREADS_PER_BLOCK     256

#ifdef AXIS_DOUBLEPRECISION
// Double-precision solver

/// <summary>
/// Defines the precision type of point coordinates used by the solver.
/// </summary>
typedef double coordtype;

/// <summary>
/// Defines the precision type of numerical results used by the solver.
/// </summary>
typedef	double real;
#else
// Single-precision solver

/// <summary>
/// Defines the precision type of point coordinates used by the solver.
/// </summary>
typedef float coordtype;

/// <summary>
/// Defines the precision type of numerical results used by the solver.
/// </summary>
typedef	float real;
#endif

/**********************************************************************************************//**
 * @typedef	long id_type
 *
 * @brief	Data type for object identifiers.
 **************************************************************************************************/
typedef long id_type;

/**********************************************************************************************//**
 * @typedef	long size_type
 *
 * @brief	Data type for length or quantity measures.
 **************************************************************************************************/
typedef long size_type;

/**********************************************************************************************//**
 * @typedef	unsigned long storage_space_type
 *
 * @brief	Data type for capacity measures.
 **************************************************************************************************/
typedef unsigned long storage_space_type;


#if defined(__GNUC__)
	#include <inttypes.h>
	typedef uint64_t uint64;
	typedef int64_t  int64;
#elif defined(__INTEL_COMPILER)
	typedef unsigned __int64 uint64;
	typedef __int64          int64;
#elif defined(_MSC_VER)
	typedef unsigned __int64 uint64;
	typedef __int64          int64;
#endif
