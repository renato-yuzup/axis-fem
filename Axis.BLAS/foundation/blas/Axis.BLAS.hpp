/// <summary>
/// Contains macro definitions used throughout the BLAS library.
/// </summary>
/// <author>Renato T. Yamassaki</author>

// The following ifdef block is the standard way of creating macros which make exporting
// from a DLL simpler. All files within this DLL are compiled with the AXISBLAS_EXPORTS
// symbol defined on the command line. This symbol should not be defined on any project
// that uses this DLL. This way any other project whose source files include this file see
// AXISBLAS_API functions as being imported from a DLL, whereas this DLL sees symbols
// defined with this macro as being exported.
#ifdef AXISBLAS_EXPORTS
	#if defined(_MSC_VER) 
		#define AXISBLAS_API __declspec(dllexport)
	#elif defined(_GCC)
		#define AXISBLAS_API __attribute__((visibility("default")))
	#endif
#else
	#if defined(_MSC_VER)
		#define AXISBLAS_API __declspec(dllimport)
	#elif defined(_GCC)
		#define AXISBLAS_API
	#endif
#endif


/// <summary>
/// Include type definitions that match the selected architecture.
/// </summary
#include "foundation/basic_types.hpp"

/// <summary>
/// The default scheduling rule for OpenMP directives
/// </summary>
// #define    SCHEDULE_DEFAULT			schedule(static, 128) /*schedule(guided,4)*/
// #define    BLAS_SCHEDULE_DEFAULT	schedule(guided,4)

#ifdef AXIS_DOUBLEPRECISION
	/// <summary>
	/// Defines the BLAS function used to multiply two matrices.
	/// </summary>
	#define matrix_general_prod					cblas_dgemm

	/// <summary>
	/// Defines the BLAS function used to factorize a matrix using LU algorithm.
	/// </summary>
	#define matrix_general_LU_factorize			dgetrf

	/// <summary>
	/// Defines the BLAS function used to solve a linear system of equations.
	/// </summary>
	#define matrix_general_solve				dgetrs
#else
	/// <summary>
	/// Defines the BLAS function used to multiply two matrices.
	/// </summary>
	#define matrix_general_prod					cblas_sgemm

	/// <summary>
	/// Defines the BLAS function used to factorize a matrix using LU algorithm.
	/// </summary>
	#define matrix_general_LU_factorize			sgetrf

	/// <summary>
	/// Defines the BLAS function used to solve a linear system of equations.
	/// </summary>
	#define matrix_general_solve				sgetrs
#endif
