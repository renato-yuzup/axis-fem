// The following ifdef block is the standard way of creating macros which make exporting 
// from a DLL simpler. All files within this DLL are compiled with the AXISYUZUTEST_EXPORTS
// symbol defined on the command line. This symbol should not be defined on any project
// that uses this DLL. This way any other project whose source files include this file see 
// AXISYUZUTEST_API functions as being imported from a DLL, whereas this DLL sees symbols
// defined with this macro as being exported.
#ifdef AXISYUZUTEST_EXPORTS
#define AXISYUZUTEST_API __declspec(dllexport)
#else
#define AXISYUZUTEST_API __declspec(dllimport)
#endif

// This class is exported from the Axis.YuzuTest.dll
class AXISYUZUTEST_API CAxisYuzuTest {
public:
	CAxisYuzuTest(void);
	// TODO: add your methods here.
};

extern AXISYUZUTEST_API int nAxisYuzuTest;

AXISYUZUTEST_API int fnAxisYuzuTest(void);
