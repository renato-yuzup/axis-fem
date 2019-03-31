// The following ifdef block is the standard way of creating macros which make exporting 
// from a DLL simpler. All files within this DLL are compiled with the AXISSTANDARDMATERIALS_EXPORTS
// symbol defined on the command line. This symbol should not be defined on any project
// that uses this DLL. This way any other project whose source files include this file see 
// AXISSTANDARDMATERIALS_API functions as being imported from a DLL, whereas this DLL sees symbols
// defined with this macro as being exported.
#ifdef AXISSTANDARDMATERIALS_EXPORTS
#define AXISSTANDARDMATERIALS_API __declspec(dllexport)
#else
#define AXISSTANDARDMATERIALS_API __declspec(dllimport)
#endif

