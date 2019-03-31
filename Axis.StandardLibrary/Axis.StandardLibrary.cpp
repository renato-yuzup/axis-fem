// axis.StandardLibrary.cpp : Defines the exported functions for the DLL application.
//

#include "stdafx.h"
#include "Axis.StandardLibrary.h"

// This is an example of an exported variable
AXISSTANDARDLIBRARY_API int nAxisStandardLibrary=0;

// This is an example of an exported function.
AXISSTANDARDLIBRARY_API int fnAxisStandardLibrary(void)
{
	return 42;
}

// This is the constructor of a class that has been exported.
// see Axis.StandardLibrary.h for the class definition
CAxisStandardLibrary::CAxisStandardLibrary()
{
	return;
}

