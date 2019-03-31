#pragma once
#include "foundation/Axis.Mint.hpp"
#include "ParseTreeNode.hpp"

namespace axis { namespace services { namespace language { namespace parsing {

class AXISMINT_API ExpressionNode : public ParseTreeNode
{
public:
	ExpressionNode(void);
	virtual ~ExpressionNode(void);

	virtual bool IsAssignment(void) const = 0;
	virtual bool IsRhs(void) const = 0;
	virtual bool IsEnumeration(void) const = 0;

	void AddChild(ParseTreeNode& node);
	const ParseTreeNode *GetFirstChild(void) const;
	const ParseTreeNode *GetLastChild(void) const;
	int GetChildCount(void) const;
	virtual bool IsEmpty(void) const;

	virtual axis::String ToString( void ) const;
	virtual axis::String ToExpressionString(void) const;

	virtual bool IsTerminal( void ) const;
private:
  enum CharType
  {
    IdentifierDeclarationType = 0,
    NonIdentifierDeclarationType = 1
  };

  void InitMembers(void);
  CharType CheckCharType( axis::String::value_type c ) const;

  ParseTreeNode *_firstChild;
  ParseTreeNode *_lastChild;
  int _childrenCount;
};			

} } } } // namespace axis::services::language::parsing
