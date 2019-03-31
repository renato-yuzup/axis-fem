#pragma once
#include "application/parsing/parsers/BlockParser.hpp"
#include "application/factories/parsers/BlockProvider.hpp"
#include "services/language/primitives/GeneralExpressionParser.hpp"
#include "services/language/primitives/OrExpressionParser.hpp"
#include "services/language/primitives/EnumerationParser.hpp"
#include "foundation/Axis.SystemBase.hpp"
#include "domain/collections/NodeSet.hpp"
#include "domain/curves/Curve.hpp"

namespace axis { namespace application { namespace parsing { namespace parsers {

class NodalLoadParser : public axis::application::parsing::parsers::BlockParser
{
public:
	NodalLoadParser(axis::application::factories::parsers::BlockProvider& parentProvider);
	~NodalLoadParser(void);
	virtual BlockParser& GetNestedContext( const axis::String& contextName, 
    const axis::services::language::syntax::evaluation::ParameterList& paramList );
	virtual axis::services::language::parsing::ParseResult Parse( 
    const axis::services::language::iterators::InputIterator& begin, 
    const axis::services::language::iterators::InputIterator& end );
private:
  void InitGrammar(void);
  void ParseNodalLoad( const axis::services::language::parsing::ParseTreeNode& parseTree );
  void ReadParseInformation(const axis::services::language::parsing::ParseTreeNode& parseTree,
    axis::String& nodeSetId, axis::String& curveId, bool *directionsEnabled, real& scaleFactor);
  bool CheckSymbols(const axis::String& nodeSetId, axis::domain::collections::NodeSet& nodeSet, 
    int dofIndex, const axis::String& curveId);
  void BuildNodalLoad(axis::domain::collections::NodeSet& nodeSet, int dofIndex, 
    axis::foundation::memory::RelativePointer& curvePtr, real scaleFactor);
  axis::String GetSymbolName(id_type nodeId, int dofIndex) const;
  bool IsNodeSetInitialized( const axis::String& nodeSetId );

	axis::application::factories::parsers::BlockProvider& _parentProvider;
	axis::services::language::primitives::OrExpressionParser _validIdentifiers;
	axis::services::language::primitives::OrExpressionParser _possibleDirections;
	axis::services::language::primitives::OrExpressionParser _directionWord;
	axis::services::language::primitives::OrExpressionParser _nodalLoadExpression;
	axis::services::language::primitives::GeneralExpressionParser _shortNodalLoadExpression;
	axis::services::language::primitives::GeneralExpressionParser _completeNodalLoadExpression;
	axis::services::language::primitives::EnumerationParser *_directionExpression;
};

} } } } // namespace axis::application::parsing::parsers
