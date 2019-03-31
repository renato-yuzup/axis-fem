/// <summary>
/// Contains definitions for macros used in the Axis Common Library.
/// </summary>
/// <author>Renato T. Yamassaki</author>

#pragma once

#include <math.h>
#define NULL				0

// The following ifdef block is the standard way of creating macros which make exporting
// from a DLL simpler. All files within this DLL are compiled with the AXISCOMMONLIBRARY_EXPORTS
// symbol defined on the command line. This symbol should not be defined on any project
// that uses this DLL. This way any other project whose source files include this file see
// AXISCOMMONLIBRARY_API functions as being imported from a DLL, whereas this DLL sees symbols
// defined with this macro as being exported.
#ifdef AXISCOMMONLIBRARY_EXPORTS
	#if defined(_MSC_VER)
		#define AXISCOMMONLIBRARY_API __declspec(dllexport)
	#elif defined(_GCC)
		#define AXISCOMMONLIBRARY_API __attribute__((visibility("default")))
	#endif
#else
	#if defined(_MSC_VER)
		#define AXISCOMMONLIBRARY_API __declspec(dllimport)
	#elif defined(_GCC)
		#define AXISCOMMONLIBRARY_API
	#endif
#endif

// Macro used to suppress export warnings in Windows systems
#ifdef _MSC_VER
#define __AXIS_SUPPRESS_EXPORT_WARNING		__pragma(warning(suppress: 4251))	// Microsoft specific
#else
#define __AXIS_SUPPRESS_EXPORT_WARNING
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



#define    SOLVER_OMP_SCHEDULE_DEFAULT			schedule(guided,4)
