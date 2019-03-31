#include "AxisException.hpp"
#include "SourceHintSet.hpp"

using namespace axis::foundation;
using namespace axis;

AxisException::AxisException(void) : _traceHints(*new SourceHintSet())
{
	_innerException = NULL;
}

AxisException::~AxisException(void)
{
	delete &_traceHints;
	if (_innerException != NULL)
	{	// delete our copied exception
		delete _innerException;
		_innerException = NULL;
	}
}

AxisException::AxisException( const String& message, const AxisException *innerException ) : _traceHints(*new SourceHintSet())
{
	_description = String(message);
	_innerException = &innerException->Clone();
}

AxisException::AxisException( const AxisException *innerException ) : _traceHints(*new SourceHintSet())
{
	_innerException = &innerException->Clone();
}

AxisException::AxisException( const String& message ) : _traceHints(*new SourceHintSet())
{
	_description = String(message);
	_innerException = NULL;
}

void axis::foundation::AxisException::SetInnerException( const AxisException *innerException )
{
	if (_innerException == innerException) return;
	if (_innerException != NULL)
	{
		delete _innerException;
	}
	_innerException = &innerException->Clone();
}

AxisException *axis::foundation::AxisException::GetInnerException( void ) const
{
	return _innerException;
}

String axis::foundation::AxisException::GetMessage( void ) const
{
	return _description;
}

void axis::foundation::AxisException::SetMessage( const String& message )
{
	_description = message;
}

axis::foundation::AxisException::AxisException( const AxisException& ex ) : _traceHints(*new SourceHintSet())
{
	_innerException = NULL;
	Copy(ex);
}

AxisException& axis::foundation::AxisException::operator=( const AxisException& ex )
{
	Copy(ex);
	return *this;
}

void axis::foundation::AxisException::AddTraceHint( const SourceTraceHint& hint )
{
	// ignore if duplicated
	if (!_traceHints.Contains(hint))
	{
		_traceHints.Add(hint);
	}
}

bool axis::foundation::AxisException::HasSourceTraceHint( const SourceTraceHint& hint ) const
{
	return _traceHints.Contains(hint);
}

AxisException& axis::foundation::AxisException::operator<<( const SourceTraceHint& hint )
{
	AddTraceHint(hint);
	return *this;
}

AxisException& axis::foundation::AxisException::Clone( void ) const
{
	return *new AxisException(*this);
}

AxisException& axis::foundation::AxisException::operator<<( const AxisException& innerException )
{
	SetInnerException(&innerException);
	return *this;
}

void axis::foundation::AxisException::Copy( const AxisException& e )
{
	if (&e == this) return;	// avoid self assignment
	_description = e._description;
	if (_innerException != NULL)
	{
		delete _innerException;
	}
	if (e._innerException != NULL)
	{
		_innerException = &e._innerException->Clone();
	}
	else	
	{
		_innerException = NULL;
	}
	_traceHints.Clear();
	e.PushHints(*this);
}

void axis::foundation::AxisException::PushHints( AxisException& e ) const
{
	SourceHintCollection::Visitor& v = _traceHints.GetVisitor();
	try
	{
		for (;v.HasNext(); v.GoNext())
		{
			e << v.GetItem();
		}
	}
	catch (...)
	{
		delete &v;
		throw;
	}
	delete &v;
}

axis::String axis::foundation::AxisException::GetTypeName( void ) const
{
	return _T("AxisException");
}