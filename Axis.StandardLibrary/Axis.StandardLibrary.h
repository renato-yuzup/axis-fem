// The following ifdef block is the standard way of creating macros which make exporting 
// from a DLL simpler. All files within this DLL are compiled with the AXISSTANDARDLIBRARY_EXPORTS
// symbol defined on the command line. This symbol should not be defined on any project
// that uses this DLL. This way any other project whose source files include this file see 
// AXISSTANDARDLIBRARY_API functions as being imported from a DLL, whereas this DLL sees symbols
// defined with this macro as being exported.
#ifdef AXISSTANDARDLIBRARY_EXPORTS
#define AXISSTANDARDLIBRARY_API __declspec(dllexport)
#else
#define AXISSTANDARDLIBRARY_API __declspec(dllimport)
#endif

// This class is exported from the Axis.StandardLibrary.dll
class AXISSTANDARDLIBRARY_API CAxisStandardLibrary {
public:
	CAxisStandardLibrary(void);
	// TODO: add your methods here.
};

extern AXISSTANDARDLIBRARY_API int nAxisStandardLibrary;

AXISSTANDARDLIBRARY_API int fnAxisStandardLibrary(void);
