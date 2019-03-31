// The following ifdef block is the standard way of creating macros which make exporting 
// from a DLL simpler. All files within this DLL are compiled with the AXISLOCALEBRAZILIANPORTUGUESE_EXPORTS
// symbol defined on the command line. This symbol should not be defined on any project
// that uses this DLL. This way any other project whose source files include this file see 
// AXISLOCALEBRAZILIANPORTUGUESE_API functions as being imported from a DLL, whereas this DLL sees symbols
// defined with this macro as being exported.
#ifdef AXISLOCALEBRAZILIANPORTUGUESE_EXPORTS
#define AXISLOCALEBRAZILIANPORTUGUESE_API __declspec(dllexport)
#else
#define AXISLOCALEBRAZILIANPORTUGUESE_API __declspec(dllimport)
#endif

// This class is exported from the Axis.Locale.BrazilianPortuguese.dll
class AXISLOCALEBRAZILIANPORTUGUESE_API CAxisLocaleBrazilianPortuguese {
public:
	CAxisLocaleBrazilianPortuguese(void);
	// TODO: add your methods here.
};

extern AXISLOCALEBRAZILIANPORTUGUESE_API int nAxisLocaleBrazilianPortuguese;

AXISLOCALEBRAZILIANPORTUGUESE_API int fnAxisLocaleBrazilianPortuguese(void);
