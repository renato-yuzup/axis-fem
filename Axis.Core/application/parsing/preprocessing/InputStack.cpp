#include "InputStack.hpp"
#include "foundation/OutOfBoundsException.hpp"
#include "foundation/InvalidOperationException.hpp"
#include "foundation/ArgumentException.hpp"

#define AXIS_PREPROCESSOR_FILE_STACK_DEPTH		256

namespace aapp = axis::application::parsing::preprocessing;
namespace asio = axis::services::io;

aapp::InputStack::InputStack(void)
{
	_stack = new asio::StreamReader*[AXIS_PREPROCESSOR_FILE_STACK_DEPTH];
	_topElement = -1;
	_maxStackLength = AXIS_PREPROCESSOR_FILE_STACK_DEPTH;
}

aapp::InputStack::~InputStack(void)
{
	// releases resources used by every stream in the stack
	for (int i = 0; i <= _topElement; i++)
	{
		_stack[i]->Close();
		delete _stack[i];
	}
	delete [] _stack;
}

int aapp::InputStack::GetMaximumSize( void ) const
{
	return _maxStackLength;
}

void aapp::InputStack::SetMaximumSize( int maxLength )
{
	if (maxLength > AXIS_PREPROCESSOR_FILE_STACK_DEPTH || maxLength <= 0)
	{
		throw axis::foundation::ArgumentException();
	}
	_maxStackLength = maxLength;
}

int aapp::InputStack::AddStream( axis::String streamDescriptor )
{
	// first, open input stream
	asio::FileReader *file = new asio::FileReader(streamDescriptor);
	// if everything went ok, push to stack
	_topElement++;
	_stack[_topElement] = file;
	return _topElement;
}

int aapp::InputStack::AddStream( asio::StreamReader& stream )
{
	_topElement++;
	_stack[_topElement] = &stream;
	return _topElement;
}

int aapp::InputStack::Count( void ) const
{
	return _topElement + 1;
}

asio::StreamReader& aapp::InputStack::GetStream( int id ) const
{
	if (id < 0 || id > _topElement)
	{
		throw axis::foundation::OutOfBoundsException();
	}
	return *_stack[id];
}

bool aapp::InputStack::CanStore( void ) const
{
	return (Count() < GetMaximumSize());
}

void aapp::InputStack::CloseTopStream( void )
{
	if (_topElement < 1)
	{
		throw axis::foundation::InvalidOperationException();
	}
	_stack[_topElement]->Close();
	delete _stack[_topElement];
	_stack[_topElement] = NULL;
	_topElement--;
}

asio::StreamReader& aapp::InputStack::GetTopStream( void ) const
{
	if (_topElement < 0)
	{
		throw axis::foundation::InvalidOperationException();
	}
	return *_stack[_topElement];
}

void aapp::InputStack::CloseNestedStreams( void )
{
	while(Count() > 1)
	{
		CloseTopStream();
	}
}
