#pragma once
#include "foundation/Axis.Mint.hpp"
#include "AxisString.hpp"

namespace axis { namespace services { namespace language { namespace parsing {

class AXISMINT_API ParseTreeNode
{
public:
	ParseTreeNode(void);
	ParseTreeNode(ParseTreeNode &parent);
	ParseTreeNode(ParseTreeNode &parent, ParseTreeNode &previousSibling);

	virtual ~ParseTreeNode(void);
	virtual axis::String ToString(void) const = 0;
	virtual axis::String ToExpressionString(void) const;
	axis::String BuildExpressionString(void) const;
	virtual bool IsTerminal(void) const = 0;
	virtual bool IsEmpty(void) const = 0;

	bool IsRoot(void) const;
	ParseTreeNode *GetParent(void) const;
	ParseTreeNode *GetPreviousSibling(void) const;
	ParseTreeNode *GetNextSibling(void) const;
	void SetParent(ParseTreeNode& parent);
	ParseTreeNode& SetNextSibling(ParseTreeNode& nextSibling);

	virtual ParseTreeNode& Clone(void) const = 0;

	void NotifyUse(void);
	void NotifyDestroy(void);
private:
	enum CharType
	{
		IdentifierDeclarationType = 0,
		NonIdentifierDeclarationType = 1
	};

  void InitMembers(void);
  CharType CheckCharType( axis::String::value_type c ) const;

	ParseTreeNode *_parent;
	ParseTreeNode *_nextSibling;
	ParseTreeNode *_previouSibling;
	size_t _useCount;
};			

} } } } // namespace axis::services::language::parsing
