#if defined DEBUG || defined _DEBUG
#include "BogusFileReader.hpp"
#include "foundation/IOException.hpp"

BogusFileReader::BogusFileReader( const axis::String& filename ) : FileReader(filename)
{
	_lineCountToFail = 2;
}


BogusFileReader::~BogusFileReader(void)
{
}

void BogusFileReader::ReadLine( axis::String& s )
{
	if (_lineCountToFail == 0)
	{	
		// simulate an I/O error
		throw axis::foundation::IOException();
	}
	--_lineCountToFail;

	s.assign(_T("BOGUS LINE"));
}

#endif