#pragma once
#include "foundation/Axis.Mint.hpp"
#include "ParserAction.hpp"

namespace axis { namespace services { namespace language { namespace actions {

class ParserCallback;

class AXISMINT_API CallbackAction : public ParserAction
{
public:
	CallbackAction(ParserCallback& callback);
	CallbackAction(ParserCallback& callback, int tag);
	CallbackAction(ParserCallback& callback, void *data);

	virtual void Run( const axis::services::language::parsing::ParseResult& result ) const;

	virtual ParserAction& Clone( void ) const;
private:
	char _callbackType;
	int _tag;
	void *_data;
	ParserCallback *_callbackObj;	// object to call to perform callback
};					

} } } } // namespace axis::services::language::actions
