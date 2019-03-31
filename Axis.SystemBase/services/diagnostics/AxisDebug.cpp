#include <assert.h>
#include "AxisDebug.hpp"
#include "foundation/AssertionFailedException.hpp"

axis::services::diagnostics::AxisDebug::AxisDebug( void )
{
	// nothing to do 
}

axis::services::diagnostics::AxisDebug::~AxisDebug( void )
{
	// nothing to do
}

void axis::services::diagnostics::AxisDebug::MarkAssertionFailed( const char * expr, const char * function, const char * file, long line )
{
	throw axis::foundation::AssertionFailedException(expr, function, file, line);
}