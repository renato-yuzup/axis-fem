#pragma once
#include "foundation/Axis.SystemBase.hpp"
#include "application/factories/elements/HexahedronFactory.hpp"
#include "application/factories/parsers/BlockProvider.hpp"
#include "application/parsing/parsers/BlockParser.hpp"
#include "application/parsing/core/SectionDefinition.hpp"
#include "domain/collections/ElementSet.hpp"
#include "domain/elements/FiniteElement.hpp"
#include "domain/elements/ElementGeometry.hpp"
#include "domain/elements/Node.hpp"
#include "services/language/primitives/GeneralExpressionParser.hpp"
#include "services/language/primitives/OrExpressionParser.hpp"

namespace axis { namespace application { namespace parsing { namespace parsers {

class HexahedronElementParser : public BlockParser
{
public:
	HexahedronElementParser(const axis::application::factories::parsers::BlockProvider& parentProvider, 
    const axis::application::parsing::core::SectionDefinition& definition, 
    axis::domain::collections::ElementSet& elementCollection,
    axis::application::factories::elements::HexahedronFactory& factory);
	~HexahedronElementParser(void);
						
	virtual BlockParser& GetNestedContext( const axis::String& contextName, 
    const axis::services::language::syntax::evaluation::ParameterList& paramList );

	virtual axis::services::language::parsing::ParseResult Parse( 
    const axis::services::language::iterators::InputIterator& begin, 
    const axis::services::language::iterators::InputIterator& end );
private:
  void InitGrammar(void);
  void ProcessElementInformation(const axis::services::language::parsing::ParseTreeNode& parseTree);
  bool ValidateElementInformation(
    const axis::services::language::parsing::ParseTreeNode& parseTree) const;
  void ExtractNodeDataFromParseTree(const axis::services::language::parsing::ParseTreeNode& parseTree, 
    id_type connectivity[]) const;
  id_type GetElementIdFromParseTree(
    const axis::services::language::parsing::ParseTreeNode& parseTree) const;
  void BuildHexahedronElement(id_type elementId, id_type connectivity[]) const;
  bool CheckNodesExistence(id_type connectivity[]) const;
  void RegisterMissingNodes(id_type connectivity[]);
  bool AreNodesCompatible( const id_type connectivity[] ) const;
  void RegisterIncompatibleNodes( const id_type connectivity[], id_type elementId );

	const axis::application::factories::parsers::BlockProvider& provider_;
	const axis::application::parsing::core::SectionDefinition& elementDefinition_;
	axis::domain::collections::ElementSet& elementSet_;
  axis::application::factories::elements::HexahedronFactory& factory_;

	// our grammar
	axis::services::language::primitives::GeneralExpressionParser elementExpression_;
	axis::services::language::primitives::GeneralExpressionParser elementIdSeparator_;
	axis::services::language::primitives::GeneralExpressionParser elementConnectivitySeparator_;
	axis::services::language::primitives::OrExpressionParser connectivitySeparator_;
	axis::services::language::primitives::OrExpressionParser idSeparator_;
};				

} } } } // namespace axis::application::parsing::parsers
