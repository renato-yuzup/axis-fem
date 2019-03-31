#include "NodeSetParser.hpp"
#include "AxisString.hpp"
#include "foundation/definitions/AxisInputLanguage.hpp"
#include "application/parsing/error_messages.hpp"
#include "application/parsing/warning_messages.hpp"
#include "domain/analyses/NumericalModel.hpp"
#include "services/language/syntax/evaluation/NumberValue.hpp"
#include "services/language/factories/AxisGrammar.hpp"
#include "services/language/parsing/ExpressionNode.hpp"
#include "services/language/parsing/NumberTerminal.hpp"
#include "application/parsing/core/SymbolTable.hpp"
#include "services/messaging/ErrorMessage.hpp"
#include "foundation/InvalidSyntaxException.hpp"
#include "foundation/CustomParserErrorException.hpp"
#include "foundation/NotSupportedException.hpp"
#include "foundation/ArgumentException.hpp"
#include "services/messaging/WarningMessage.hpp"

namespace aafp = axis::application::factories::parsers;
namespace aapc = axis::application::parsing::core;
namespace aapps = axis::application::parsing::parsers;
namespace ada = axis::domain::analyses;
namespace ade = axis::domain::elements;
namespace adc = axis::domain::collections;
namespace aslf = axis::services::language::factories;
namespace asli = axis::services::language::iterators;
namespace aslp = axis::services::language::parsing;
namespace aslse = axis::services::language::syntax::evaluation;
namespace asmm = axis::services::messaging;
namespace afd = axis::foundation::definitions;

template<class Char, class Traits>
std::basic_ostream<Char,Traits> & operator <<(
	std::basic_ostream<Char,Traits> &o, const axis::String &s)
{
	return o << s.c_str();
}

template<class Char, class Traits>
std::basic_istream<Char,Traits> & operator >>(
	std::basic_istream<Char,Traits> &o, const axis::String &s)
{
	Char c;
	o >> c;
	s.append((String::value_type)c);
	return o;
}

aapps::NodeSetParser::NodeSetParser(aafp::BlockProvider& factory, const aslse::ParameterList& paramList) : 
  _parentProvider(factory), _paramList(paramList.Clone()), 
  _nodeIdentifier(aslf::AxisGrammar::CreateNumberParser())
{
	_mustIgnoreParse = false;
	InitGrammar();
}

void aapps::NodeSetParser::InitGrammar( void )
{
	_nodeRange << aslf::AxisGrammar::CreateNumberParser() 
             << aslf::AxisGrammar::CreateOperatorParser(_T("-")) 
             << aslf::AxisGrammar::CreateNumberParser();
}

aapps::NodeSetParser::~NodeSetParser(void)
{
	delete &_paramList;
}

aapps::BlockParser& aapps::NodeSetParser::GetNestedContext( const axis::String& contextName, 
                                                            const aslse::ParameterList& paramList )
{
	// check if the parent provider knows any provider that can handle it
	if (!_parentProvider.ContainsProvider(contextName, paramList))
	{
		throw axis::foundation::NotSupportedException();
	}

	aafp::BlockProvider& subProvider = _parentProvider.GetProvider(contextName, paramList);
	return subProvider.BuildParser(contextName, paramList);
}

aslp::ParseResult aapps::NodeSetParser::Parse(const asli::InputIterator& begin, 
                                              const asli::InputIterator& end)
{
	aslp::ParseResult resultRange = _nodeRange(begin, end);
	aslp::ParseResult resultUnique = _nodeIdentifier(begin, end);
	bool errorFlag = false;

	if (resultRange.IsMatch())
	{
		// get range values
		aslp::ExpressionNode& root = (aslp::ExpressionNode&)resultRange.GetParseTree();
    const aslp::NumberTerminal* firstTerm = ((const aslp::NumberTerminal*)root.GetFirstChild());
    const aslp::NumberTerminal* lastTerm  = ((const aslp::NumberTerminal*)root.GetLastChild());
    id_type fromId = firstTerm->GetInteger();
		id_type toId   = lastTerm->GetInteger();

		// validate range
		if (!firstTerm->IsInteger())
		{
			String s = AXIS_ERROR_MSG_INVALID_VALUE_TYPE;
			s.append(firstTerm->ToString());
			GetParseContext().RegisterEvent(asmm::ErrorMessage(0x300503, s));
			errorFlag = true;
		}
		if (!lastTerm->IsInteger())
		{
			String s = AXIS_ERROR_MSG_INVALID_VALUE_TYPE;
			s.append(lastTerm->ToString());
			GetParseContext().RegisterEvent(asmm::ErrorMessage(0x300503, s));
			errorFlag = true;
		}
		if (fromId <= 0)
		{
			String s = AXIS_ERROR_MSG_VALUE_OUT_OF_RANGE;
			s.append(String::int_parse(fromId));
			GetParseContext().RegisterEvent(asmm::ErrorMessage(0x300504, s));
			errorFlag = true;
		}
		if (toId <= 0)
		{
			String s = AXIS_ERROR_MSG_VALUE_OUT_OF_RANGE;
			s.append(String::int_parse(toId));
			GetParseContext().RegisterEvent(asmm::ErrorMessage(0x300504, s));
			errorFlag = true;
		}
		if (fromId > toId)
		{
			String s = AXIS_ERROR_MSG_NODESET_PARSER_INVALID_RANGE;
			s.append(String::int_parse(fromId));
			GetParseContext().RegisterEvent(asmm::ErrorMessage(0x300504, s));
			errorFlag = true;
		}
		if (!errorFlag)
		{
			AddToNodeSet(fromId, toId);
		}
		return resultRange;
	}
	else if(resultRange.GetResult() == aslp::ParseResult::FullReadPartialMatch && 
         (resultUnique.GetLastReadPosition() < resultRange.GetLastReadPosition()))
	{
		return resultRange;
	}
	else if (resultUnique.IsMatch())
	{
		// get range values
    aslp::ExpressionNode& root = (aslp::ExpressionNode&)resultUnique.GetParseTree();
    aslp::NumberTerminal& term = ((aslp::NumberTerminal&)root);
		id_type id = term.GetInteger();

		// validate range
		if (!term.IsInteger())
		{
			String s = AXIS_ERROR_MSG_INVALID_VALUE_TYPE;
			s.append(term.ToString());
			GetParseContext().RegisterEvent(asmm::ErrorMessage(0x300503, s));
			errorFlag = true;
		}
		if (id <= 0)
		{
			String s = AXIS_ERROR_MSG_VALUE_OUT_OF_RANGE;
			s.append(String::int_parse(id));
			GetParseContext().RegisterEvent(asmm::ErrorMessage(0x300504, s));
			errorFlag = true;
		}
		else if(!errorFlag)
		{
			AddToNodeSet(id, id);
		}
	}

	return resultUnique;
}

void aapps::NodeSetParser::DoStartContext( void )
{
  aapc::SymbolTable& st = GetParseContext().Symbols();
	ada::NumericalModel& model = GetAnalysis().GetNumericalModel();

	// check parameter list correctness
	if (_paramList.IsDeclared(afd::AxisInputLanguage::NodeSetSyntax.SetIdAttributeName))
	{
		aslse::ParameterValue& paramValue = 
      _paramList.GetParameterValue(afd::AxisInputLanguage::NodeSetSyntax.SetIdAttributeName);
		_nodeSetAlias = paramValue.ToString();
		if (!paramValue.IsAtomic())
		{
			String s = AXIS_ERROR_MSG_INVALID_BLOCK_PARAM_TYPE;
			s.append(afd::AxisInputLanguage::NodeSetSyntax.SetIdAttributeName);
			GetParseContext().RegisterEvent(asmm::ErrorMessage(0x300505, s));
			_mustIgnoreParse = true;
		}
		aslse::AtomicValue& atomicVal = (aslse::AtomicValue&)paramValue;
		if (!(atomicVal.IsId() || atomicVal.IsString()))
		{
			if (atomicVal.IsNumeric())
			{	// it's ok only if it is a non-negative integer
				aslse::NumberValue& numVal = (aslse::NumberValue&)atomicVal;
				if (!(numVal.IsInteger() && numVal.GetLong() >= 0))
				{
					String s = AXIS_ERROR_MSG_INVALID_BLOCK_PARAM_TYPE;
					s.append(afd::AxisInputLanguage::NodeSetSyntax.SetIdAttributeName);
					GetParseContext().RegisterEvent(asmm::ErrorMessage(0x300505, s));
					_mustIgnoreParse = true;
				}
			}
			else
			{
				String s = AXIS_ERROR_MSG_INVALID_BLOCK_PARAM_TYPE;
				s.append(afd::AxisInputLanguage::NodeSetSyntax.SetIdAttributeName);
				GetParseContext().RegisterEvent(asmm::ErrorMessage(0x300505, s));
				_mustIgnoreParse = true;
			}
		}

		String nodeSetId = _paramList.GetParameterValue(
      afd::AxisInputLanguage::NodeSetSyntax.SetIdAttributeName).ToString();

		if (model.ExistsNodeSet(nodeSetId))
		{	// couldn't add node set because already exists; depending on the circumstances, this is bad
			if (st.IsSymbolCurrentRoundDefined(nodeSetId, aapc::SymbolTable::kNodeSet))
			{	// this is bad; two equally named node sets
				String s = AXIS_ERROR_MSG_NODESET_PARSER_DUPLICATED_ID;
				s.append(nodeSetId);
				GetParseContext().RegisterEvent(asmm::ErrorMessage(0x30051B, s));
				_mustIgnoreParse = true;
			}
			else
			{	// nah, we are just re-reading the previous node set; refresh symbol
				st.DefineOrRefreshSymbol(nodeSetId, aapc::SymbolTable::kNodeSet);
			}
			_isFirstTimeRead = false;
		}
		else
		{	// it's a new node set, mark it as pending to add
			_nodeSet = new adc::NodeSet();
			_isFirstTimeRead = true;
		}

		_paramList.Consume(afd::AxisInputLanguage::NodeSetSyntax.SetIdAttributeName);
	}

	// init state variables
	_hasUnresolvedNodes = false;

	// if we still have parameters not processed, warn user
	if (_paramList.Count() > 0) _mustIgnoreParse = true;
	WarnUnrecognizedParams(_paramList);
}

void aapps::NodeSetParser::AddToNodeSet( id_type from, id_type to )
{
	bool hasOverlapping = false;
	aapc::SymbolTable& st = GetParseContext().Symbols();
  ada::NumericalModel& model = GetAnalysis().GetNumericalModel();
	adc::NodeSet& defaultNodeSet = model.Nodes();

	// ignore if we are just re-reading it
	if (!_isFirstTimeRead) return;

	for (id_type i = from; i <= to; ++i)
	{
		if (!defaultNodeSet.IsUserIndexed(i))
		{
			if (GetParseContext().GetRunMode() != aapc::ParseContext::kInspectionMode)
			{
				// node does not exist; create cross reference
				st.AddCurrentRoundUnresolvedSymbol(String::int_parse(i), aapc::SymbolTable::kNode);
			}
			else
			{	// node not found
				String s = AXIS_ERROR_MSG_NODE_NOT_FOUND;
				s.append(String::int_parse(i));
				GetParseContext().RegisterEvent(asmm::ErrorMessage(0x300506, s));
			}
			_hasUnresolvedNodes = true;
		}
		else
		{	// add node to the set, if it is already there, just warn it
			if (!_mustIgnoreParse)
			{
				if (_nodeSet->IsUserIndexed(i))
				{
					hasOverlapping = true;
				}
				else
				{
					_nodeSet->Add(defaultNodeSet.GetPointerByUserId(i));
				}
			}
		}
	}
	if (hasOverlapping)
	{
		GetParseContext().RegisterEvent(
      asmm::WarningMessage(0x30051C, AXIS_WARN_MSG_NODESET_PARSER_DOUBLE_ADD));
	}
}

void aapps::NodeSetParser::DoCloseContext( void )
{
	aapc::SymbolTable& st = GetParseContext().Symbols();
  ada::NumericalModel& model = GetAnalysis().GetNumericalModel();

	// add node set to the analysis object, if required and
	// conditions make it acceptable
	if (_isFirstTimeRead)
	{
		if (!_hasUnresolvedNodes && !_mustIgnoreParse)
		{	// ok, it is a complete node set -- add it
			model.AddNodeSet(_nodeSetAlias, *_nodeSet);
			st.DefineOrRefreshSymbol(_nodeSetAlias, aapc::SymbolTable::kNodeSet);
		}
		else
		{	// we couldn't build a complete node set -- discard it and wait
			// for a next round until we can build it complete
			delete _nodeSet;
		}
	}
}
