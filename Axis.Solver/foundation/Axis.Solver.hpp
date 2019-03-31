// The following ifdef block is the standard way of creating macros which make exporting 
// from a DLL simpler. All files within this DLL are compiled with the AXISSOLVER_EXPORTS
// symbol defined on the command line. This symbol should not be defined on any project
// that uses this DLL. This way any other project whose source files include this file see 
// AXISSOLVER_API functions as being imported from a DLL, whereas this DLL sees symbols
// defined with this macro as being exported.
#ifdef AXISSOLVER_EXPORTS
	#if defined(_MSC_VER) 
		#define AXISSOLVER_API __declspec(dllexport)
	#elif defined(_GCC)
		#define AXISSOLVER_API __attribute__((visibility("default")))
	#endif
	#define AXISSOLVER_STL_EXPORTS
#else
	#if defined(_MSC_VER)
		#define AXISSOLVER_API __declspec(dllimport)
	#elif defined(_GCC)
		#define AXISSOLVER_API
	#endif
	#define AXISSOLVER_STL_EXPORTS extern
#endif

#ifdef _MSC_VER
#define __AXISSOLVER_SUPPRESS_EXPORT_WARNING		__pragma(warning(suppress: 4251))	// Microsoft specific
#else
#define __AXISSOLVER_SUPPRESS_EXPORT_WARNING		
#endif

#define    SOLVER_OMP_SCHEDULE_DEFAULT			schedule(guided,4)
