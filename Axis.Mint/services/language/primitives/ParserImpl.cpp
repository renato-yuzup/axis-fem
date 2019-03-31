#include "ParserImpl.hpp"
#include "../actions/NopAction.hpp"
#include "../parsing/ExpressionNode.hpp"
#include "foundation/ArgumentException.hpp"
#include "foundation/NotSupportedException.hpp"

namespace asla = axis::services::language::actions;
namespace aslpp = axis::services::language::primitives;

aslpp::ParserImpl::ParserImpl( void )
{
	refCount_ = 0;
	action_ = new actions::NopAction();
}

aslpp::ParserImpl::~ParserImpl( void )
{	// delete our action
	delete action_;
}

void aslpp::ParserImpl::NotifyDestroy( void )
{
	--refCount_;
	if (refCount_ <= 0)
	{	// no more object is using this object; remove it
		delete this;
	}
}

void aslpp::ParserImpl::NotifyUse( void )
{
	++refCount_;
}

void aslpp::ParserImpl::AddAction( const asla::ParserAction& action )
{
	// create new action
	asla::ParserAction *newAction = &action.Clone();
	
	// delete and replace old action
	delete action_;
	action_ = newAction;
}

aslpp::ParserImpl& aslpp::ParserImpl::operator<<( const asla::ParserAction& action )
{
	AddAction(action);
	return *this;
}

