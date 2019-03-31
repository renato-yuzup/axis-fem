#include "CallbackAction.hpp"
#include "ParserCallback.hpp"

namespace {
const char CallbackType_None = 0;
const char CallbackType_WithTag = 1;
const char CallbackType_WithData = 2;
} // namespace

namespace asla = axis::services::language::actions;
namespace aslp = axis::services::language::parsing;

asla::CallbackAction::CallbackAction( ParserCallback& callback )
{
	_callbackType = CallbackType_None;
	_callbackObj = &callback;
}

asla::CallbackAction::CallbackAction( ParserCallback& callback, int tag )
{
	_callbackType = CallbackType_WithTag;
	_callbackObj = &callback;
	_tag = tag;
}

asla::CallbackAction::CallbackAction( ParserCallback& callback, void *data )
{
	_callbackType = CallbackType_WithData;
	_callbackObj = &callback;
	_data = data;
}

void asla::CallbackAction::Run( const aslp::ParseResult& result ) const
{
	switch(_callbackType)
	{
	case CallbackType_None:
		_callbackObj->ProcessLexerSuccessEvent(result);
		break;
	case CallbackType_WithTag:
		_callbackObj->ProcessLexerSuccessEvent(result, _tag);
		break;
	case CallbackType_WithData:
		_callbackObj->ProcessLexerSuccessEvent(result, _data);
		break;
	}
}

asla::ParserAction& asla::CallbackAction::Clone( void ) const
{
	switch(_callbackType)
	{
	case CallbackType_None:
		return *new CallbackAction(*_callbackObj);
		break;
	case CallbackType_WithTag:
		return *new CallbackAction(*_callbackObj, _tag);
		break;
	default:
		return *new CallbackAction(*_callbackObj, _data);
		break;
	}
}