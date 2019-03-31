#include "ElementParser.hpp"

#include "application/factories/parsers/BlockProvider.hpp"
#include "application/parsing/error_messages.hpp"
#include "application/parsing/core/ParseContext.hpp"
#include "application/parsing/core/SectionDefinition.hpp"
#include "application/parsing/core/Sketchbook.hpp"
#include "application/parsing/core/SymbolTable.hpp"

#include "domain/analyses/NumericalModel.hpp"
#include "domain/collections/ElementSet.hpp"

#include "foundation/definitions/AxisInputLanguage.hpp"
#include "foundation/NotSupportedException.hpp"
#include "foundation/InvalidOperationException.hpp"

#include "services/messaging/ErrorMessage.hpp"

namespace aal = axis::application::locators;
namespace aapc = axis::application::parsing::core;
namespace aapps = axis::application::parsing::parsers;
namespace ada = axis::domain::analyses;
namespace adc = axis::domain::collections;
namespace asli = axis::services::language::iterators;
namespace aslp = axis::services::language::parsing;
namespace aslse = axis::services::language::syntax::evaluation;
namespace asmm = axis::services::messaging;
namespace afd = axis::foundation::definitions;

aapps::ElementParser::ElementParser( aal::ElementParserLocator& parentProvider, 
                                     const aslse::ParameterList& paramList ) :
_parentProvider(parentProvider)
{
	_elementSetId = 
    paramList.GetParameterValue(afd::AxisInputLanguage::ElementSyntax.SetIdAttributeName).ToString();
	_innerParser = NULL;
}

aapps::ElementParser::~ElementParser(void)
{
	if (_innerParser != NULL)
	{
		delete _innerParser;
		_innerParser = NULL;
	}
}

aapps::BlockParser& aapps::ElementParser::GetNestedContext( const axis::String& contextName, 
                                                            const aslse::ParameterList& paramList )
{
	if (_parentProvider.ContainsProvider(contextName, paramList))
	{
		axis::application::factories::parsers::BlockProvider& provider = _parentProvider.GetProvider(contextName, paramList);
		BlockParser& nestedContext = provider.BuildParser(contextName, paramList);
		nestedContext.SetAnalysis(GetAnalysis());
		return nestedContext;
	}

	// no provider found
	throw axis::foundation::NotSupportedException();
}

aslp::ParseResult aapps::ElementParser::Parse(const asli::InputIterator& begin, 
                                              const asli::InputIterator& end)
{
	if (_innerParser == NULL)
	{
		throw axis::foundation::InvalidOperationException();
	}
	return _innerParser->Parse(begin, end);
}

void aapps::ElementParser::DoCloseContext( void )
{
	// finalize parser
	if (_innerParser != NULL)
	{
		_innerParser->CloseContext();
		_innerParser->DetachFromAnalysis();

		delete _innerParser;
		_innerParser = NULL;
	}
}

void aapps::ElementParser::DoStartContext( void )
{
  aapc::SymbolTable& st = GetParseContext().Symbols();
  aapc::Sketchbook& sketchbook = GetParseContext().Sketches();

	// check if there is a declared section definition
	ada::NumericalModel& model = GetAnalysis().GetNumericalModel();
	if (sketchbook.HasSectionDefined(_elementSetId))
	{	// there is, get an appropriable parser
		const aapc::SectionDefinition& section = sketchbook.GetSection(_elementSetId);
		if (!_parentProvider.CanBuildElement(section))
		{	// we can't build the element -- unknown section definition; trigger an error
			TriggerError(AXIS_ERROR_MSG_PART_PARSER_INVALID_SECTION, 0x300530);
		}
		else
		{	// create our worker parser
			adc::ElementSet& elementSet = EnsureElementSet(_elementSetId);
			_innerParser = &_parentProvider.BuildParser(section, elementSet);
		}
	}
	else
	{	// mark section as pending adding a cross-ref
		if (GetParseContext().GetRunMode() == aapc::ParseContext::kInspectionMode)
		{	// definition not found
			String s = AXIS_ERROR_MSG_PART_NOT_FOUND;
			s += _elementSetId;
			TriggerError(s, 0x300531);
		}
		else
		{	// probably section not defined yet -- warn it
			st.AddCurrentRoundUnresolvedSymbol(_elementSetId, aapc::SymbolTable::kSection);
		}
	}

	// if we couldn't get a valid parser, create a void parser
	if (_innerParser == NULL)
	{
		_innerParser = &_parentProvider.BuildVoidParser();
	}

	// initialize worker parser
	_innerParser->SetAnalysis(GetAnalysis());
	_innerParser->StartContext(GetParseContext());
}

void aapps::ElementParser::TriggerError( const axis::String& errorMsg, const int errorId ) const
{
	GetParseContext().RegisterEvent(asmm::ErrorMessage(errorId, errorMsg));
}

adc::ElementSet& aapps::ElementParser::EnsureElementSet( const axis::String& elementSetId ) const
{
  aapc::SymbolTable& st = GetParseContext().Symbols();
	ada::NumericalModel& model = GetAnalysis().GetNumericalModel();
	if (model.ExistsElementSet(elementSetId))
	{
		return model.GetElementSet(elementSetId);
	}
	// create and return a new element set
	adc::ElementSet& set = *new adc::ElementSet();
	model.AddElementSet(elementSetId, set);
	// notify element set creation
	st.DefineOrRefreshSymbol(elementSetId, aapc::SymbolTable::kElementSet);
	return set;
}
