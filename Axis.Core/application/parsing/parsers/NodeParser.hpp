#pragma once
#include "application/factories/parsers/BlockProvider.hpp"
#include "application/factories/elements/NodeFactory.hpp"
#include "services/language/primitives/GeneralExpressionParser.hpp"
#include "services/language/primitives/OrExpressionParser.hpp"
#include "services/language/syntax/evaluation/ParameterList.hpp"
#include "services/language/parsing/NumberTerminal.hpp"
#include "foundation/Axis.SystemBase.hpp"
#include "domain/collections/NodeSet.hpp"

namespace axis { namespace application { namespace parsing { namespace parsers {

class NodeParser : public axis::application::parsing::parsers::BlockParser
{
public:
	NodeParser(axis::application::factories::parsers::BlockProvider& factory, 
    axis::application::factories::elements::NodeFactory& nodeFactory, 
    const axis::services::language::syntax::evaluation::ParameterList& params);
	virtual ~NodeParser(void);
	virtual axis::application::parsing::parsers::BlockParser& GetNestedContext(
    const axis::String& contextName, 
    const axis::services::language::syntax::evaluation::ParameterList& paramList);
	virtual axis::services::language::parsing::ParseResult Parse(
    const axis::services::language::iterators::InputIterator& begin, 
    const axis::services::language::iterators::InputIterator& end);
protected:
	virtual void DoStartContext( void );
private:
  bool ExtractAndValidateNodeData(axis::services::language::parsing::ParseTreeNode& parseTree, 
    id_type& nodeId, double& nodeX, double& nodeY, double& nodeZ ) const;

  axis::services::language::primitives::GeneralExpressionParser _nodeExpression;
	axis::services::language::primitives::GeneralExpressionParser _nodeIdSeparatorType1;
	axis::services::language::primitives::GeneralExpressionParser _nodeIdSeparatorType2;
	axis::services::language::primitives::OrExpressionParser _nodeIdSeparator;
	axis::services::language::primitives::OrExpressionParser _nodeCoordSeparator;
	bool _mustIgnoreParse;
	axis::services::language::syntax::evaluation::ParameterList& _paramList;
	axis::application::factories::parsers::BlockProvider& _parentProvider;
	axis::application::factories::elements::NodeFactory& _nodeFactory;
	axis::domain::collections::NodeSet *_currentNodeSet;

};							

} } } } // namespace axis::application::parsing::parsers
