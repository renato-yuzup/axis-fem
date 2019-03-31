#include "ElementSetParser.hpp"
#include <boost/lexical_cast.hpp>

#include "foundation/InvalidSyntaxException.hpp"
#include "foundation/CustomParserErrorException.hpp"
#include "foundation/NotSupportedException.hpp"
#include "foundation/ArgumentException.hpp"
#include "AxisString.hpp"

#include "services/language/syntax/evaluation/NumberValue.hpp"
#include "services/language/factories/AxisGrammar.hpp"
#include "services/language/parsing/ExpressionNode.hpp"
#include "services/language/parsing/NumberTerminal.hpp"

#include "domain/analyses/NumericalModel.hpp"
#include "application/parsing/error_messages.hpp"
#include "application/parsing/warning_messages.hpp"
#include "application/parsing/core/SymbolTable.hpp"
#include "foundation/definitions/AxisInputLanguage.hpp"
#include "services/messaging/ErrorMessage.hpp"

namespace aapc = axis::application::parsing::core;
namespace aapps = axis::application::parsing::parsers;
namespace aafp = axis::application::factories::parsers;
namespace af = axis::foundation;
namespace afd = axis::foundation::definitions;
namespace ada = axis::domain::analyses;
namespace ade = axis::domain::elements;
namespace adc = axis::domain::collections;
namespace aslf = axis::services::language::factories;
namespace asli = axis::services::language::iterators;
namespace aslp = axis::services::language::parsing;
namespace aslse = axis::services::language::syntax::evaluation;
namespace asmm = axis::services::messaging;

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

aapps::ElementSetParser::ElementSetParser(aafp::BlockProvider& factory,
	                                        const aslse::ParameterList& paramList) :
_parentProvider(factory), _paramList(paramList.Clone()), 
_elementIdentifier(aslf::AxisGrammar::CreateNumberParser())
{
	_mustIgnoreParse = false;
	InitGrammar();
}

void aapps::ElementSetParser::InitGrammar( void )
{
	_elementRange << aslf::AxisGrammar::CreateNumberParser() 
                << aslf::AxisGrammar::CreateOperatorParser(_T("-")) 
                << aslf::AxisGrammar::CreateNumberParser();
}

aapps::ElementSetParser::~ElementSetParser(void)
{
	delete &_paramList;
}

aapps::BlockParser& aapps::ElementSetParser::GetNestedContext( const axis::String& contextName, 
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

aslp::ParseResult aapps::ElementSetParser::Parse(const asli::InputIterator& begin, 
                                                 const asli::InputIterator& end)
{
	aslp::ParseResult resultRange = _elementRange(begin, end);
	aslp::ParseResult resultUnique = _elementIdentifier(begin, end);
	bool errorFlag = false;

	if (resultRange.IsMatch())
	{
		// get range values
		aslp::ExpressionNode& root = (aslp::ExpressionNode&)resultRange.GetParseTree();
    const aslp::NumberTerminal* firstTerm = ((const aslp::NumberTerminal*)root.GetFirstChild());
    const aslp::NumberTerminal* lastTerm  = ((const aslp::NumberTerminal*)root.GetLastChild());
		id_type fromId                        = firstTerm->GetInteger();
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
			GetParseContext().RegisterEvent(asmm::ErrorMessage(0x300503, s));
			errorFlag = true;
		}
		if (toId <= 0)
		{
			String s = AXIS_ERROR_MSG_VALUE_OUT_OF_RANGE;
			s.append(String::int_parse(toId));
			GetParseContext().RegisterEvent(asmm::ErrorMessage(0x300503, s));
			errorFlag = true;
		}
		if (fromId > toId)
		{
			String s = AXIS_ERROR_MSG_NODESET_PARSER_INVALID_RANGE;
			s.append(String::int_parse(fromId));
			GetParseContext().RegisterEvent(asmm::ErrorMessage(0x300503, s));
			errorFlag = true;
		}
		if (!errorFlag)
		{
			AddToElementSet(fromId, toId);
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
			AddToElementSet(id, id);
		}
	}

	return resultUnique;
}

void aapps::ElementSetParser::DoStartContext( void )
{
	aapc::SymbolTable& st = GetParseContext().Symbols();
  ada::NumericalModel& model = GetAnalysis().GetNumericalModel();

	// check parameter list correctness
	if (_paramList.IsDeclared(afd::AxisInputLanguage::ElementSetSyntax.SetIdAttributeName))
	{
		aslse::ParameterValue& paramValue = 
        _paramList.GetParameterValue(afd::AxisInputLanguage::ElementSetSyntax.SetIdAttributeName);
		_elementSetAlias = paramValue.ToString();
		if (!paramValue.IsAtomic())
		{
			String s = AXIS_ERROR_MSG_INVALID_BLOCK_PARAM_TYPE;
			s.append(afd::AxisInputLanguage::ElementSetSyntax.SetIdAttributeName);
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
					s.append(afd::AxisInputLanguage::ElementSetSyntax.SetIdAttributeName);
					GetParseContext().RegisterEvent(asmm::ErrorMessage(0x300505, s));
					_mustIgnoreParse = true;
				}
			}
			else
			{
				String s = AXIS_ERROR_MSG_INVALID_BLOCK_PARAM_TYPE;
				s.append(afd::AxisInputLanguage::ElementSetSyntax.SetIdAttributeName);
				GetParseContext().RegisterEvent(asmm::ErrorMessage(0x300505, s));
				_mustIgnoreParse = true;
			}
		}

		String elementSetId = _paramList.GetParameterValue(
      afd::AxisInputLanguage::ElementSetSyntax.SetIdAttributeName).ToString();

		if (model.ExistsElementSet(elementSetId))
		{	// couldn't add element set because it already exists
			if (st.IsSymbolCurrentRoundDefined(elementSetId, aapc::SymbolTable::kElementSet))
			{
				String s = AXIS_ERROR_MSG_ELEMENTSET_PARSER_DUPLICATED_ID;
				s.append(elementSetId);
				GetParseContext().RegisterEvent(asmm::ErrorMessage(0x300516, s));
				_mustIgnoreParse = true;
			}
			else
			{	// we are just passing by the same element set we defined -- ignore
				st.DefineOrRefreshSymbol(elementSetId, aapc::SymbolTable::kElementSet);
			}
			_isFirstTimeRead = false;
		}
		else
		{	// it's a new element set, mark it as pending to add
			_elementSet = new adc::ElementSet();
			_isFirstTimeRead = true;
		}

		_paramList.Consume(afd::AxisInputLanguage::ElementSetSyntax.SetIdAttributeName);
	}

	// init state variables
	_hasUnresolvedElements = false;

	// if we still have parameters not processed, warn user
	if (_paramList.Count() > 0) _mustIgnoreParse = true;
	WarnUnrecognizedParams(_paramList);
}

void aapps::ElementSetParser::AddToElementSet( id_type from, id_type to )
{
	aapc::SymbolTable& st = GetParseContext().Symbols();
  ada::NumericalModel& model = GetAnalysis().GetNumericalModel();
	adc::ElementSet& defaultElementSet = model.Elements();
  bool hasOverlapping = false;

	// ignore if we are just re-reading the same element set
	if (!_isFirstTimeRead) return;

	for (id_type i = from; i <= to; ++i)
	{
		if (!defaultElementSet.IsUserIndexed(i))
		{
			if (GetParseContext().GetRunMode() != aapc::ParseContext::kInspectionMode)
			{
				// element does not exist; create cross reference
				st.AddCurrentRoundUnresolvedSymbol(String::int_parse(i), aapc::SymbolTable::kElement);
			}
			else
			{	// element not found
				String s = AXIS_ERROR_MSG_ELEMENT_NOT_FOUND;
				s.append(String::int_parse(i));
				GetParseContext().RegisterEvent(asmm::ErrorMessage(0x300517, s));
			}
			_hasUnresolvedElements = true;
		}
		else
		{	// add element to the set, if it is already there, just warn it
			if (!_mustIgnoreParse)
			{
				if (_elementSet->IsUserIndexed(i))
				{
					hasOverlapping = true;
				}
				else
				{
					_elementSet->Add(defaultElementSet.GetPointerByUserId(i));
				}
			}
		}
	}
	if (hasOverlapping)
	{
		GetParseContext().RegisterEvent(
      asmm::ErrorMessage(0x200501, AXIS_WARN_MSG_ELEMENTSET_PARSER_DOUBLE_ADD));
	}
}

void aapps::ElementSetParser::DoCloseContext( void )
{
  aapc::SymbolTable& st = GetParseContext().Symbols();
  // add element set to the analysis object, if required and
	// conditions make it acceptable
	if (_isFirstTimeRead)
	{
		if (!_hasUnresolvedElements && !_mustIgnoreParse)
		{	// ok, it is a complete node set -- add it
			GetAnalysis().GetNumericalModel().AddElementSet(_elementSetAlias, *_elementSet);
			st.DefineOrRefreshSymbol(_elementSetAlias, aapc::SymbolTable::kElementSet);
		}
		else
		{	// we couldn't build a complete element set -- discard it and wait
			// for a next round until we can build it complete
			delete _elementSet;
		}
	}
}
