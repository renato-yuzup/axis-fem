#include "ExpressionList.hpp"
#include "foundation/ArgumentException.hpp"

namespace aslpp = axis::services::language::primitives;

aslpp::ExpressionList::ExpressionList( void )
{
	InitMembers();
}

aslpp::ExpressionList::~ExpressionList( void )
{
	// drop all list nodes, without effectively deleting its items
	Clear();
}

void aslpp::ExpressionList::InitMembers( void )
{
	_firstNode = NULL;
	_lastNode = NULL;
	_count = 0;
}

void aslpp::ExpressionList::Add( const ExpressionParser& expression )
{
	// create node and append to the end of the list
	ExpressionListNode *node = new ExpressionListNode(expression, _lastNode);
	if (_count == 0)	// we are inserting the first list item
	{
		_firstNode = node;
		_lastNode = node;
	}
	else
	{	// second and onwards node
		_lastNode->Next = node;
		_lastNode = node;
	}
	++_count;
}

void aslpp::ExpressionList::Remove( const ExpressionParser& expression )
{
	ExpressionListNode *node = GetNode(expression);
	if (node == NULL)
	{	// node not found
		throw axis::foundation::ArgumentException();
	}
	
	// destroy node and re-link adjacent list items
	if (node->Previous != NULL)
	{
		node->Previous->Next = node->Next;
	}
	if (node->Next != NULL)
	{
		node->Next->Previous = node->Previous;
	}
	if (_lastNode == node)
	{
		_lastNode = node->Previous;
	}
	if (_firstNode == node)
	{
		_firstNode = node->Next;
	}
	delete node;
	--_count;
}

bool aslpp::ExpressionList::Contains( const ExpressionParser& expression ) const
{
	return (GetNode(expression) != NULL);
}

bool aslpp::ExpressionList::IsEmpty( void ) const
{
	return (_count == 0);
}

void aslpp::ExpressionList::Clear( void )
{
	// drop and delete all nodes
	ExpressionListNode *node = _firstNode;
	while (node != NULL)
	{
		ExpressionListNode *nextNode = node->Next;
		delete node;
		node = nextNode;
	}
	InitMembers();
}

void aslpp::ExpressionList::ClearAndDestroy( void )
{
	// drop and delete all nodes, destroying its items along
	ExpressionListNode *node = _firstNode;
	while (node != NULL)
	{
		ExpressionListNode *nextNode = node->Next;
		delete &node->Item;
		delete node;
		node = nextNode;
	}
	InitMembers();
}

size_t aslpp::ExpressionList::Count( void ) const
{
	return _count;
}

aslpp::ExpressionList::ExpressionListNode * aslpp::ExpressionList::GetNode( 
  const ExpressionParser& expression ) const
{
	ExpressionListNode *node = _firstNode;
	while (node != NULL)
	{
		if (&node->Item == &expression)
		{
			return node;
		}
		node = node->Next;
	}

	// not found
	return NULL;
}

aslpp::ExpressionList::Iterator aslpp::ExpressionList::First( void ) const
{
	return Iterator(*_firstNode);
}

aslpp::ExpressionList::Iterator aslpp::ExpressionList::Last( void ) const
{
	return Iterator(*_lastNode);
}

/*                           ========= SUBCLASS CONSTRUCTORS =========                            */
aslpp::ExpressionList::ExpressionListNode::ExpressionListNode( const ExpressionParser& expression ) : 
  Item(expression)
{
	Previous = NULL;
	Next = NULL;
}

aslpp::ExpressionList::ExpressionListNode::ExpressionListNode( const ExpressionParser& expression, 
                                                               ExpressionListNode *previous ) : 
Item(expression)
{
	Previous = previous;
	Next = NULL;
}

aslpp::ExpressionList::ExpressionListNode::ExpressionListNode( const ExpressionParser& expression, 
                                                               ExpressionListNode *previous, 
                                                               ExpressionListNode *next ) : 
Item(expression)
{
	Previous = previous;
	Next = next;
}

aslpp::ExpressionList::Iterator::Iterator( void )
{
	_node = NULL;
}

aslpp::ExpressionList::Iterator::Iterator( const ExpressionListNode& node )
{
	_node = &node;
}

aslpp::ExpressionList::Iterator::Iterator( const Iterator& iterator )
{
	_node = iterator._node;
}

aslpp::ExpressionList::Iterator aslpp::ExpressionList::Iterator::operator++( void )
{
	_node = _node->Next;
	return *this;
}

aslpp::ExpressionList::Iterator aslpp::ExpressionList::Iterator::operator++( int )
{
	Iterator it(*this);
	++*this;
	return it;
}

const aslpp::ExpressionParser *aslpp::ExpressionList::Iterator::operator->( void ) const
{
	return &_node->Item;
}

const aslpp::ExpressionParser& aslpp::ExpressionList::Iterator::operator*( void ) const
{
	return _node->Item;
}

bool aslpp::ExpressionList::Iterator::IsValid( void ) const
{
	return _node != NULL;
}

aslpp::ExpressionList::Iterator aslpp::ExpressionList::Iterator::operator--( void )
{
	_node = _node->Previous;
	return *this;
}

aslpp::ExpressionList::Iterator aslpp::ExpressionList::Iterator::operator--( int )
{
	Iterator it(*this);
	_node = _node->Previous;
	return it;
}

bool aslpp::ExpressionList::Iterator::operator==( const Iterator& it ) const
{
	return _node == it._node;
}

bool aslpp::ExpressionList::Iterator::operator!=( const Iterator& it ) const
{
	return !(*this == it);
}
