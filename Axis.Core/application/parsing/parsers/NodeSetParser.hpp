#ifndef __BASICNODESETPARSER_HPP
#define __BASICNODESETPARSER_HPP

#include "application/factories/parsers/BlockProvider.hpp"
#include "application/parsing/parsers/BlockParser.hpp"
#include "services/language/primitives/GeneralExpressionParser.hpp"
#include "services/language/primitives/AtomicExpressionParser.hpp"
#include "domain/collections/NodeSet.hpp"
#include "foundation/Axis.SystemBase.hpp"

namespace axis { namespace application { namespace parsing { namespace parsers {

class NodeSetParser : public axis::application::parsing::parsers::BlockParser
{
public:
	NodeSetParser(axis::application::factories::parsers::BlockProvider& factory, 
    const axis::services::language::syntax::evaluation::ParameterList& paramList);
	virtual ~NodeSetParser(void);
	virtual axis::application::parsing::parsers::BlockParser& GetNestedContext(
    const axis::String& contextName, 
    const axis::services::language::syntax::evaluation::ParameterList& paramList);
	virtual axis::services::language::parsing::ParseResult Parse(
    const axis::services::language::iterators::InputIterator& begin, 
    const axis::services::language::iterators::InputIterator& end);
protected:
	virtual void DoStartContext(void);
	virtual void DoCloseContext( void );
private:
  void InitGrammar(void);
  void AddToNodeSet(id_type from, id_type to);

	axis::application::factories::parsers::BlockProvider& _parentProvider;
	axis::services::language::syntax::evaluation::ParameterList& _paramList;
	bool _mustIgnoreParse;
	bool _isFirstTimeRead;
	bool _hasUnresolvedNodes;
	axis::domain::collections::NodeSet  *_nodeSet;
	axis::String _nodeSetAlias;
	axis::services::language::primitives::GeneralExpressionParser _nodeRange;
	axis::services::language::primitives::AtomicExpressionParser _nodeIdentifier;
};

} } } } // namespace axis::application::parsing::parsers

#endif