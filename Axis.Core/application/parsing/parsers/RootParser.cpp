#include "RootParser.hpp"
#include "foundation/NotSupportedException.hpp"
#include "foundation/InvalidSyntaxException.hpp"
#include "foundation/CustomParserErrorException.hpp"
#include "foundation/UnexpectedExpressionException.hpp"
#include "services/language/factories/AxisGrammar.hpp"
#include "services/language/parsing/SymbolTerminal.hpp"
#include "services/language/parsing/RhsExpression.hpp"
#include "domain/analyses/NumericalModel.hpp"

namespace aafp  = axis::application::factories::parsers;
namespace aapps = axis::application::parsing::parsers;
namespace asli  = axis::services::language::iterators;
namespace aslf  = axis::services::language::factories;
namespace asli  = axis::services::language::iterators;
namespace aslp  = axis::services::language::parsing;
namespace aslse = axis::services::language::syntax::evaluation;
namespace af    = axis::foundation;

aapps::RootParser::RootParser( aafp::BlockProvider& factory ) : 
	_parent(factory)
{
	InitGrammar();
}

aapps::RootParser::~RootParser( void )
{
	// do nothing	
}

void aapps::RootParser::InitGrammar( void )
{
	_titleExpression << _expression1 << _expression2;
	_expression1     << aslf::AxisGrammar::CreateReservedWordParser(_T("TITLE")) 
                   << aslf::AxisGrammar::CreateStringParser(false);
	_expression2     << aslf::AxisGrammar::CreateReservedWordParser(_T("TITLE")) << _titleSeparator 
                   << aslf::AxisGrammar::CreateStringParser(false);
	_titleSeparator  << aslf::AxisGrammar::CreateOperatorParser(_T("=")) 
                   << aslf::AxisGrammar::CreateOperatorParser(_T(":")) 
                   << aslf::AxisGrammar::CreateReservedWordParser(_T("IS"));
}

aapps::BlockParser& aapps::RootParser::GetNestedContext( const axis::String& contextName, 
                                                         const aslse::ParameterList& paramList )
{
	if (_parent.ContainsProvider(contextName, paramList))
	{
		aafp::BlockProvider& provider = _parent.GetProvider(contextName, paramList);
		aapps::BlockParser& nestedContext = provider.BuildParser(contextName, paramList);
		nestedContext.SetAnalysis(GetAnalysis());
		return nestedContext;
	}

	// no provider found
	throw axis::foundation::NotSupportedException();
}

aslp::ParseResult aapps::RootParser::Parse(const asli::InputIterator& begin, 
                                           const asli::InputIterator& end)
{
	aslp::ParseResult result = _titleExpression(begin, end);
	if (result.IsMatch())
	{	// found an analysis title declaration
		String analysisTitle;
		const aslp::ParseTreeNode *node = 
      static_cast<const aslp::RhsExpression&>(result.GetParseTree()).GetFirstChild();
		node = node->GetNextSibling();
		const aslp::SymbolTerminal& term = static_cast<const aslp::SymbolTerminal&>(*node);

		// check if after the first token if we have a string (omitted
		// operator) or not
		if (term.IsString())	// assignment operator was omitted
		{
			analysisTitle = term.ToString();
		}
		else
		{	// jump to next token and get the string
			analysisTitle = node->GetNextSibling()->ToString();
		}
		// assign analysis title
		GetAnalysis().SetTitle(analysisTitle);
	}
	return result;
}
