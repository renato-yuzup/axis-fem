#pragma once
#include "foundation/Axis.Mint.hpp"
#include "ExpressionParser.hpp"

namespace axis { namespace services { namespace language { namespace primitives {

class AXISMINT_API ExpressionList
{
public:
	class ExpressionListNode
	{
	public:
		ExpressionListNode *Previous;
		ExpressionListNode *Next;
		const ExpressionParser &Item;

		ExpressionListNode(const ExpressionParser& expression);
		ExpressionListNode(const ExpressionParser& expression, ExpressionListNode *previous);
		ExpressionListNode(const ExpressionParser& expression, ExpressionListNode *previous, ExpressionListNode *next);
	};
	class Iterator
	{
	private:
		const ExpressionListNode *_node;
	public:
		Iterator(void);
		Iterator(const ExpressionListNode& node);
		Iterator(const Iterator& iterator);

		Iterator operator ++(void);
		Iterator operator ++(int);

		Iterator operator --(void);
		Iterator operator --(int);

		const ExpressionParser *operator->(void) const;
		const ExpressionParser& operator*(void) const;

		bool IsValid(void) const;

		bool operator ==(const Iterator& it) const;
		bool operator !=(const Iterator& it) const;
	};

  ExpressionList(void);
	~ExpressionList(void);

	void Add(const ExpressionParser& expression);
	void Remove(const ExpressionParser& expression);
	bool Contains(const ExpressionParser& expression) const;
	void Clear(void);
	void ClearAndDestroy(void);
	size_t Count(void) const;
	bool IsEmpty(void) const;

	Iterator First(void) const;
	Iterator Last(void) const;
private:
  void InitMembers(void);
  ExpressionListNode *GetNode(const ExpressionParser& expression) const;

	ExpressionListNode *_firstNode;
	ExpressionListNode *_lastNode;
	size_t _count;
};				

} } } } // namespace axis::services::language::primitives
