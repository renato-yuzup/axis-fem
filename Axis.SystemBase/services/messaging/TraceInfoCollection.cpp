#include "TraceInfoCollection.hpp"
#include "foundation/InvalidOperationException.hpp"

namespace asmm = axis::services::messaging;

/**************************************************************************************************************************/
/************************************************ StackNode IMPLEMENTATION ************************************************/
asmm::TraceInfoCollection::StackNode::StackNode( const TraceInfo& info ) : _traceInfo(info)
{
	_next = NULL;
	_previous = NULL;
}

asmm::TraceInfoCollection::StackNode::StackNode( const TraceInfo& info, self& next ) : _traceInfo(info)
{
	_next = &next;
	_previous = NULL;
}

asmm::TraceInfoCollection::StackNode::StackNode( const TraceInfo& info, self& next, self& previous ) : _traceInfo(info)
{
	_next = &next;
	_previous = &previous;
}

void asmm::TraceInfoCollection::StackNode::ChainNext( self& next )
{
	_next = &next;
}

void asmm::TraceInfoCollection::StackNode::ChainPrevious( self& previous )
{
	_previous = &previous;
}

const asmm::TraceInfoCollection::StackNode::self * asmm::TraceInfoCollection::StackNode::Next( void ) const
{
	return _next;
}

asmm::TraceInfoCollection::StackNode::self * asmm::TraceInfoCollection::StackNode::Next( void )
{
	return _next;
}

const asmm::TraceInfoCollection::StackNode::self * asmm::TraceInfoCollection::StackNode::Previous( void ) const
{
	return _previous;
}

asmm::TraceInfoCollection::StackNode::self * asmm::TraceInfoCollection::StackNode::Previous( void )
{
	return _previous;
}

const asmm::TraceInfoCollection::StackNode::value_type& asmm::TraceInfoCollection::StackNode::Value( void ) const
{
	return _traceInfo;
}

/************************************************************************************************************************************/
/************************************************ TraceInfoCollection IMPLEMENTATION ************************************************/
asmm::TraceInfoCollection::TraceInfoCollection( void )
{
	_first = NULL;
	_last = NULL;
	_count = 0;
}

asmm::TraceInfoCollection::TraceInfoCollection( const TraceInfoCollection& other )
{
	_first = NULL;
	_last = NULL;
	_count = 0;
	Copy(other);
}

asmm::TraceInfoCollection::~TraceInfoCollection( void )
{
	// free heap resources before destroying itself
	Clear();
}

void asmm::TraceInfoCollection::AddTraceInfo( const TraceInfo& info )
{
	StackNode *node = new StackNode(info);
	if (_last == NULL)
	{
		_last = node;
	}
	if (_first == NULL)
	{
		_first = node;
	}
	else
	{
		node->ChainNext(*_first);
		_first->ChainPrevious(*node);
		_first = node;
	}
	++_count;
}

asmm::TraceInfoCollection::value_type asmm::TraceInfoCollection::PopInfo( void )
{
	if (Empty())
	{
		throw axis::foundation::InvalidOperationException();
	}

	// clone info tag
	value_type info = _first->Value();

	// delete first node
	node_type *node = _first;
	_first = _first->Next();
	if (_first == NULL) _last = NULL;
	--_count;
	delete node;

	return info;
}

const asmm::TraceInfoCollection::value_type& asmm::TraceInfoCollection::PeekInfo( void ) const
{
	if (Empty())
	{
		throw axis::foundation::InvalidOperationException();
	}
	return _first->Value();
}

void asmm::TraceInfoCollection::Clear( void )
{
	while (_first != NULL)
	{
		node_type * node = _first;
		_first = _first->Next();
		--_count;
		delete node;
	}
	_last = NULL;
}

size_type asmm::TraceInfoCollection::Count( void ) const
{
	return _count;
}

bool asmm::TraceInfoCollection::Empty( void ) const
{
	return _count == 0;
}

asmm::TraceInfoCollection& asmm::TraceInfoCollection::operator=( const TraceInfoCollection& other )
{
	Copy(other);
	return *this;
}

void asmm::TraceInfoCollection::Copy( const TraceInfoCollection& other )
{
	// erase items before copying contents
	Clear();

	node_type *node = other._last;
	while (node != NULL)
	{
		AddTraceInfo(node->Value());
		node = node->Previous();
	}
}

bool asmm::TraceInfoCollection::Contains( int sourceId ) const
{
  node_type *node = _first;
  while (node != NULL)
  {
    const asmm::TraceInfo& info = node->Value();
    if (info.SourceId() == sourceId)
    {
      return true;
    }
    node = node->Next();
  }
  return false;
}

bool asmm::TraceInfoCollection::Contains(int sourceId, const axis::String& sourceName) const
{
  node_type *node = _first;
  while (node != NULL)
  {
    const asmm::TraceInfo& info = node->Value();
    if (info.SourceId() == sourceId && info.SourceName() == sourceName)
    {
      return true;
    }
    node = node->Next();
  }
  return false;
}

bool asmm::TraceInfoCollection::Contains(const axis::String& sourceName) const
{
  node_type *node = _first;
  while (node != NULL)
  {
    const asmm::TraceInfo& info = node->Value();
    if (info.SourceName() == sourceName)
    {
      return true;
    }
    node = node->Next();
  }
  return false;
}
