#include "HexahedronElementParser.hpp"

#include "foundation/NotSupportedException.hpp"
#include "foundation/ArgumentException.hpp"

#include "services/language/factories/AxisGrammar.hpp"
#include "services/language/parsing/ExpressionNode.hpp"
#include "services/language/parsing/SymbolTerminal.hpp"
#include "services/language/parsing/NumberTerminal.hpp"
#include "services/messaging/ErrorMessage.hpp"

#include "domain/analyses/NumericalModel.hpp"
#include "domain/elements/FiniteElement.hpp"
#include "domain/integration/IntegrationPoint.hpp"

#include "application/parsing/error_messages.hpp"
#include "application/parsing/core/ParseContext.hpp"
#include "application/parsing/core/SymbolTable.hpp"

#include "foundation/memory/pointer.hpp"

namespace aafe = axis::application::factories::elements;
namespace aafp = axis::application::factories::parsers;
namespace aapc = axis::application::parsing::core;
namespace aapps = axis::application::parsing::parsers;
namespace ada = axis::domain::analyses;
namespace adc = axis::domain::collections;
namespace ade = axis::domain::elements;
namespace aslf = axis::services::language::factories;
namespace asli = axis::services::language::iterators;
namespace aslp = axis::services::language::parsing;
namespace aslse = axis::services::language::syntax::evaluation;
namespace asmm = axis::services::messaging;
namespace afm = axis::foundation::memory;

aapps::HexahedronElementParser::HexahedronElementParser( const aafp::BlockProvider& parentProvider, 
                                                         const aapc::SectionDefinition& definition, 
                                                         adc::ElementSet& elementCollection,
                                                         aafe::HexahedronFactory& factory) :
provider_(parentProvider), elementDefinition_(definition), 
elementSet_(elementCollection), factory_(factory)
{
  InitGrammar();
}

aapps::HexahedronElementParser::~HexahedronElementParser( void )
{
  // nothing to do here
}

void aapps::HexahedronElementParser::InitGrammar( void )
{
  // initialize grammar
  elementExpression_ <<
    aslf::AxisGrammar::CreateBlankParser()  <<										        // leading spaces
    aslf::AxisGrammar::CreateNumberParser() << idSeparator_ <<						// element ID
    aslf::AxisGrammar::CreateNumberParser() << connectivitySeparator_ <<	// node 1 ID
    aslf::AxisGrammar::CreateNumberParser() << connectivitySeparator_ <<	// node 2 ID
    aslf::AxisGrammar::CreateNumberParser() << connectivitySeparator_ <<	// node 3 ID
    aslf::AxisGrammar::CreateNumberParser() << connectivitySeparator_ <<	// node 4 ID
    aslf::AxisGrammar::CreateNumberParser() << connectivitySeparator_ <<	// node 5 ID
    aslf::AxisGrammar::CreateNumberParser() << connectivitySeparator_ <<	// node 6 ID
    aslf::AxisGrammar::CreateNumberParser() << connectivitySeparator_ <<	// node 7 ID
    aslf::AxisGrammar::CreateNumberParser() <<										        // node 8 ID
    aslf::AxisGrammar::CreateBlankParser();											          // trailing spaces

  elementIdSeparator_ << aslf::AxisGrammar::CreateBlankParser() 
                      << aslf::AxisGrammar::CreateOperatorParser(_T(":")) 
                      << aslf::AxisGrammar::CreateBlankParser();
  elementConnectivitySeparator_ << aslf::AxisGrammar::CreateBlankParser() 
                                << aslf::AxisGrammar::CreateOperatorParser(_T(",")) 
                                << aslf::AxisGrammar::CreateBlankParser();

  idSeparator_ << elementIdSeparator_ 
               << elementConnectivitySeparator_ 
               << aslf::AxisGrammar::CreateBlankParser(true);
  connectivitySeparator_ << elementConnectivitySeparator_ 
                         << aslf::AxisGrammar::CreateBlankParser(true);
}

aapps::BlockParser& aapps::HexahedronElementParser::GetNestedContext( 
  const axis::String& contextName, const aslse::ParameterList& paramList )
{
  // check if the parent provider knows any provider that can handle it
  if (!provider_.ContainsProvider(contextName, paramList))
  {
    throw axis::foundation::NotSupportedException();
  }

  aafp::BlockProvider& subProvider = provider_.GetProvider(contextName, paramList);
  return subProvider.BuildParser(contextName, paramList);
}

aslp::ParseResult aapps::HexahedronElementParser::Parse( const asli::InputIterator& begin, 
                                                         const asli::InputIterator& end )
{
  aslp::ParseResult result = elementExpression_(begin, end, false);
  if (result.IsMatch())
  {
    ProcessElementInformation(result.GetParseTree());
  }
  else if (result.GetResult() == aslp::ParseResult::FailedMatch)
  {	// invalid syntax
    GetParseContext().RegisterEvent(
        asmm::ErrorMessage(0x300533, AXIS_ERROR_MSG_ELEMENT_PARSER_INVALID_DECLARATION));
  }
  return result;
}

void aapps::HexahedronElementParser::ProcessElementInformation( const aslp::ParseTreeNode& parseTree )
{
  aapc::SymbolTable& st = GetParseContext().Symbols();

  // check if parse tree is syntactically correct
  if (!ValidateElementInformation(parseTree))
  {
    GetParseContext().RegisterEvent(
          asmm::ErrorMessage(0x300533, AXIS_ERROR_MSG_ELEMENT_PARSER_INVALID_DECLARATION));
    return;
  }

  // get element id
  id_type elementId = GetElementIdFromParseTree(parseTree);

  // check element uniqueness
  ada::NumericalModel& model = GetAnalysis().GetNumericalModel();
  if (model.Elements().IsUserIndexed(elementId))
  {	// this might happen if we have already processed in a previous round
    if (st.IsSymbolCurrentRoundDefined(String::int_parse(elementId), aapc::SymbolTable::kElement))
    {	// whoops, it seems we are redefining an element defined in the current round, too bad
      String s = AXIS_ERROR_MSG_ELEMENT_PARSER_DUPLICATED_ID;
      s.append(String::int_parse(elementId));
      GetParseContext().RegisterEvent(asmm::ErrorMessage(0x300532, s));
    }
    else
    {	// we are reading the same char range, just refresh identifier and ignore processing
      st.DefineOrRefreshSymbol(String::int_parse(elementId), aapc::SymbolTable::kElement);
    }
    return;
  }

  // retrieve connectivity information
  id_type connectivity[8];
  ExtractNodeDataFromParseTree(parseTree, connectivity);

  // check if all nodes were already built
  if (!CheckNodesExistence(connectivity))
  {
    // register missing references and delay element creation
    RegisterMissingNodes(connectivity);
    return;
  }

  // check if every node has compatible dof's
  if (!AreNodesCompatible(connectivity))
  {
    RegisterIncompatibleNodes(connectivity, elementId);
    return;
  }

  // ok, we have everything we needed; start building element
  BuildHexahedronElement(elementId, connectivity);
}

bool aapps::HexahedronElementParser::ValidateElementInformation(
  const aslp::ParseTreeNode& parseTree) const
{
  bool valuesCorrect;

  // get correct nodes and check values
  const aslp::NumberTerminal *idNode  = 
      (const aslp::NumberTerminal *)((const aslp::ExpressionNode&)parseTree).GetFirstChild();
  valuesCorrect = idNode->IsInteger();

  const aslp::NumberTerminal *connectivityNode = idNode;
  for (int i = 0; i < 8; i++)
  {
    connectivityNode = (const aslp::NumberTerminal *)(connectivityNode->GetNextSibling()->IsTerminal()?
                       connectivityNode->GetNextSibling() :
                       connectivityNode->GetNextSibling()->GetNextSibling());
    valuesCorrect &= connectivityNode->IsInteger();
  }

  return valuesCorrect;
}

void aapps::HexahedronElementParser::ExtractNodeDataFromParseTree(const aslp::ParseTreeNode& parseTree, 
                                                                  id_type connectivity[] ) const
{
  const aslp::NumberTerminal *node = 
      (const aslp::NumberTerminal *)((const aslp::ExpressionNode&)parseTree).GetFirstChild();
  for (int i = 0; i < 8; i++)
  {
    node = (const aslp::NumberTerminal *)(node->GetNextSibling()->IsTerminal()?
           node->GetNextSibling() : node->GetNextSibling()->GetNextSibling());
    connectivity[i] = (id_type)node->GetInteger();
  }
}

id_type aapps::HexahedronElementParser::GetElementIdFromParseTree(
  const aslp::ParseTreeNode& parseTree) const
{
  const aslp::NumberTerminal *node = (const aslp::NumberTerminal *)
    ((const aslp::ExpressionNode&)parseTree).GetFirstChild();
  return (id_type)node->GetInteger();
}

void aapps::HexahedronElementParser::BuildHexahedronElement( id_type elementId, 
                                                             id_type connectivity[] ) const
{
  aapc::SymbolTable& st = GetParseContext().Symbols();
  ada::NumericalModel& model = GetAnalysis().GetNumericalModel();

  afm::RelativePointer ptr = factory_.BuildElement(elementId, model, connectivity, elementDefinition_);
  ade::FiniteElement& element = absref<ade::FiniteElement>(ptr);
  model.Elements().Add(ptr);
  elementSet_.Add(ptr);

  // notify creation so that if someone else was waiting for this item...
  String elementIdStr = String::int_parse(elementId);
  st.DefineOrRefreshSymbol(elementIdStr, aapc::SymbolTable::kElement);
  for (int i = 0; i < 8; i++)
  {
    String nodeIdStr = String::int_parse(connectivity[i]);
    st.DefineOrRefreshSymbol(nodeIdStr, aapc::SymbolTable::kNodeDof);
  }
}

bool aapps::HexahedronElementParser::CheckNodesExistence( id_type connectivity[] ) const
{
  ada::NumericalModel& model = GetAnalysis().GetNumericalModel();

  // verify existence of each node
  const adc::NodeSet& nodes = model.Nodes();
  for (int i = 0; i < 8; i++)
  {
    if (!nodes.IsUserIndexed(connectivity[i])) return false;
  }
  return true;
}

void aapps::HexahedronElementParser::RegisterMissingNodes( id_type connectivity[] )
{
  ada::NumericalModel& model = GetAnalysis().GetNumericalModel();

  // verify existence of each node
  for (int i = 0; i < 8; i++)
  {
    id_type nodeId = connectivity[i];
    if (!model.Nodes().IsUserIndexed(nodeId))
    {
      // if we are in inspection mode, all missing references should have
      // cleared, but it seems something has got wrong
      if (GetParseContext().GetRunMode() == aapc::ParseContext::kInspectionMode)
      {	// permanent missing nodes; trigger an error
        String nodeIdStr = String::int_parse(nodeId);
        String s = AXIS_ERROR_MSG_NODE_NOT_FOUND;
        s.append(nodeIdStr);
        GetParseContext().RegisterEvent(asmm::ErrorMessage(0x300506, s));
      }
      else
      {	// node not found; add to cross-ref table
        String nodeIdStr = String::int_parse(nodeId);
        aapc::SymbolTable& st = GetParseContext().Symbols();
        st.AddCurrentRoundUnresolvedSymbol(nodeIdStr, aapc::SymbolTable::kNode);
      }
    }
  }
}

bool aapps::HexahedronElementParser::AreNodesCompatible( const id_type *connectivity ) const
{
  ada::NumericalModel& model = GetAnalysis().GetNumericalModel();
  for (int i = 0; i < 8; i++)
  {
    int numDof = model.Nodes().GetByUserIndex(connectivity[i]).GetDofCount();
    if (!(numDof == 0 || numDof == 3)) return false;
  }
  return true;
}

void aapps::HexahedronElementParser::RegisterIncompatibleNodes( const id_type *connectivity, 
                                                                id_type elementId )
{
  ada::NumericalModel& model = GetAnalysis().GetNumericalModel();
  adc::NodeSet& nodes = model.Nodes();
  for (int i = 0; i < 8; i++)
  {
    int numDof = nodes.GetByUserIndex(connectivity[i]).GetDofCount();
    if (!(numDof == 0 || numDof == 3))
    {	// incompatible node found
      String s = AXIS_ERROR_MSG_ELEMENT_PARSER_INCOMPATIBLE_NODE;
      s = s.replace(_T("%1"), String::int_parse(connectivity[i]))
           .replace(_T("%2"), String::int_parse(elementId));
      GetParseContext().RegisterEvent(
        asmm::ErrorMessage(AXIS_ERROR_ID_ELEMENT_PARSER_INCOMPATIBLE_NODE, s));
    }
  }
}
