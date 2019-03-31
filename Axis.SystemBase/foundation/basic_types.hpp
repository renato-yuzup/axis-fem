#pragma once

#ifndef NULL
#define NULL					0
#endif

/// <summary>
/// Defines the default character type used internally by the String class.
/// Default to wchar_t (UTF-16).
/// </summary>
#ifdef _UNICODE
typedef wchar_t char_type;
#else
typedef char char_type;
#endif

/// <summary>
/// Defines the byte type so that one can easily distinguish between the char type
/// usage for string manipulation and for numerical manipulation.
/// </summary>
// #define byte char  //	typedef char byte;

/* Defines the solver precision */
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

#define MATH_ABS fabs
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

#define MATH_ABS fabsf
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