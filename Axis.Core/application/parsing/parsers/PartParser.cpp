#include "PartParser.hpp"
#include "AxisString.hpp"
#include "foundation/definitions/AxisInputLanguage.hpp"
#include "foundation/ArgumentException.hpp"
#include "foundation/InvalidSyntaxException.hpp"
#include "foundation/UnexpectedExpressionException.hpp"
#include "foundation/CustomParserErrorException.hpp"
#include "foundation/NotSupportedException.hpp"

#include "services/language/syntax/evaluation/ArrayValue.hpp"
#include "services/language/factories/AxisGrammar.hpp"
#include "services/language/grammar_tokens.hpp"
#include "services/language/parsing/ExpressionNode.hpp"
#include "services/language/parsing/IdTerminal.hpp"
#include "services/language/syntax/evaluation/ParameterList.hpp"
#include "services/language/syntax/ParameterListParser.hpp"
#include "services/messaging/ErrorMessage.hpp"

#include "domain/materials/MaterialModel.hpp"
#include "domain/collections/ElementSet.hpp"
#include "domain/analyses/NumericalModel.hpp"

#include "application/parsing/error_messages.hpp"
#include "application/parsing/core/SymbolTable.hpp"
#include "application/parsing/core/Sketchbook.hpp"

namespace aafp  = axis::application::factories::parsers;
namespace aal   = axis::application::locators;
namespace aapc  = axis::application::parsing::core;
namespace aapps = axis::application::parsing::parsers;
namespace ada   = axis::domain::analyses;
namespace adc   = axis::domain::collections;
namespace adm   = axis::domain::materials;
namespace aslf  = axis::services::language::factories;
namespace asli  = axis::services::language::iterators;
namespace aslpm = axis::services::language::primitives;
namespace aslp  = axis::services::language::parsing;
namespace asls  = axis::services::language::syntax;
namespace aslse = axis::services::language::syntax::evaluation;
namespace asmm  = axis::services::messaging;
namespace afd   = axis::foundation::definitions;

// explicit instantiate default template class
template class aapps::PartParserTemplate<aal::ElementParserLocator, aal::MaterialFactoryLocator>;

template <class ElementLocator, class MaterialLocator>
aapps::PartParserTemplate<ElementLocator, MaterialLocator>::PartParserTemplate( 
  aafp::BlockProvider& parent, ElementLocator& elementProvider, 
  MaterialLocator& constitutiveModelProvider, const aslse::ParameterList& paramList ) : 
_parent(parent), _materialProvider(constitutiveModelProvider), _elementProvider(elementProvider), 
_paramList(_enumParam, true), _arrayEnum(_arrayVal), _sectionParameters(paramList.Clone())
{
	// create parse rules
	InitGrammar();
	_sectionDefinition = NULL;

	// NOTE: validation of element settings are postponed until a call to DoStartContext is made.
}

template <class ElementLocator, class MaterialLocator>
void aapps::PartParserTemplate<ElementLocator, MaterialLocator>::InitGrammar( void )
{
	/*
		These guys we are going to use to parse the following grammar:

		paramList ::= enumParam [,enumParam]* | epsilon
		enumParam ::= assign | id
		assign ::= id = value
		value ::= id | num | str | array
		array ::= ( arrayContents )
		arrayContents ::= arrayEnum | epsilon
		arrayEnum ::= arrayVal [,arrayVal]* | epsilon
		arrayVal ::= assign | value
	*/
	_enumParam     << _assign << aslf::AxisGrammar::CreateIdParser();
	_assign.SetRhsExpression(_value);
	_value         << aslf::AxisGrammar::CreateIdParser() << aslf::AxisGrammar::CreateNumberParser() 
                 << aslf::AxisGrammar::CreateStringParser() << _array;
	_array         << aslf::AxisGrammar::CreateOperatorParser(AXIS_GRAMMAR_ARRAY_OPEN_DELIMITER)
		             << _arrayContents
		             << aslf::AxisGrammar::CreateOperatorParser(AXIS_GRAMMAR_ARRAY_CLOSE_DELIMITER);
	_arrayContents << _arrayEnum << aslf::AxisGrammar::CreateEpsilonParser();
	_arrayVal      << _assign << _value;

	_elementSetId  << aslf::AxisGrammar::CreateIdParser() << aslf::AxisGrammar::CreateNumberParser();

	// this one parses simple material expressions
	_materialExpression << aslf::AxisGrammar::CreateReservedWordParser(_T("SET")) 
                      << _elementSetId << aslf::AxisGrammar::CreateReservedWordParser(_T("IS")) 
                      << aslf::AxisGrammar::CreateIdParser() 
                      << aslf::AxisGrammar::CreateReservedWordParser(_T("WITH")) 
                      << _paramList;
}

template <class ElementLocator, class MaterialLocator>
aapps::PartParserTemplate<ElementLocator, MaterialLocator>::~PartParserTemplate(void)
{
	// delete section information
	if (_sectionDefinition != NULL)
	{
		delete _sectionDefinition;
	}
	_sectionParameters.Destroy();
}

template <class ElementLocator, class MaterialLocator>
aapps::BlockParser& aapps::PartParserTemplate<ElementLocator, MaterialLocator>::GetNestedContext( 
  const axis::String& contextName, const aslse::ParameterList& paramList )
{
	if (_parent.ContainsProvider(contextName, paramList))
	{
		return _parent.GetProvider(contextName, paramList).BuildParser(contextName, paramList);
	}
	throw axis::foundation::NotSupportedException();
}

template <class ElementLocator, class MaterialLocator>
aslp::ParseResult aapps::PartParserTemplate<ElementLocator, MaterialLocator>::Parse(
  const asli::InputIterator& begin, const asli::InputIterator& end)
{
  // this is the only syntax we accept in a (multi-)line context: a material description which
	// associates to an element set
	aslp::ParseResult result = _materialExpression(begin, end);

	if (result.IsMatch())
	{	// ok, every information needed is here, let's parse
    aapc::SymbolTable& st = GetParseContext().Symbols();
		aslp::ExpressionNode& rootNode = (aslp::ExpressionNode&)result.GetParseTree();

		// get element set id and material type id
		aslp::IdTerminal& elementSetIdTerm = 
        (aslp::IdTerminal&)*rootNode.GetFirstChild()->GetNextSibling();// after the 'SET' keyword
		aslp::IdTerminal& materialTypeTerm = 
        (aslp::IdTerminal&)*elementSetIdTerm.GetNextSibling()->GetNextSibling();	// after the 'IS' keyword
		String elementSetId = elementSetIdTerm.ToString();
		String materialTypeId = materialTypeTerm.ToString();

		// get material parameters
		aslp::EnumerationExpression& materialParametersTree = 
        (aslp::EnumerationExpression&)*materialTypeTerm.GetNextSibling()->GetNextSibling();
		aslse::ParameterList& materialParameters = 
      asls::ParameterListParser::ParseParameterList(materialParametersTree);

		// ensure element set doesn't have any section or material defined in this round
		if (st.IsSymbolCurrentRoundDefined(elementSetId, aapc::SymbolTable::kSection))
		{	// cannot redefine element section
			String s = AXIS_ERROR_MSG_PART_PARSER_SECTION_REDEFINITION;
			s = s.replace(_T("%1"), elementSetId);
			GetParseContext().RegisterEvent(asmm::ErrorMessage(0x30053C, s));
		}
		else
		{
			// check if the material description is valid
			if (_materialProvider.CanBuild(materialTypeId, materialParameters))
			{	// ok, build material (if not in error recovery mode)
				if (_sectionDefinition != NULL)
				{
					// clone section description
					aapc::SectionDefinition& section = _sectionDefinition->Clone();

					// build and attach material
					adm::MaterialModel& material = BuildMaterial(materialTypeId, materialParameters);
					section.SetMaterial(material);

					// add section definition to sketchbook if we are not in a new round
					if (!st.IsSymbolDefined(elementSetId, aapc::SymbolTable::kSection))	
					{ // note: here we check if the symbol was declared, even in a previous round
						GetParseContext().Sketches().AddSection(elementSetId, section);
					}

					// mark section as defined
					st.DefineOrRefreshSymbol(elementSetId, aapc::SymbolTable::kSection);

					// ensure element set exists
					adc::ElementSet& elementSet = EnsureElementSet(elementSetId);
				}
			}
			else
			{	// invalid material
				String s = AXIS_ERROR_MSG_PART_PARSER_INVALID_MATERIAL;
				GetParseContext().RegisterEvent(asmm::ErrorMessage(0x30053D, s));
			}
		}
	}
	return result;
}

template <class ElementLocator, class MaterialLocator>
aapc::SectionDefinition * 
  aapps::PartParserTemplate<ElementLocator, MaterialLocator>::BuildElementDescription( 
  const aslse::ParameterList& paramList ) const
{
	bool hasErrors = false;

	// check if the required parameters were declared
	if (!paramList.IsDeclared(afd::AxisInputLanguage::PartSyntax.ElementTypeParameterName))
	{	// write error
		String s = AXIS_ERROR_MSG_MISSING_BLOCK_PARAM;
		s.append(afd::AxisInputLanguage::PartSyntax.ElementTypeParameterName);
		GetParseContext().RegisterEvent(asmm::ErrorMessage(0x300515, s));
		hasErrors = true;
	}
	if (!paramList.IsDeclared(afd::AxisInputLanguage::PartSyntax.ElementDescriptionParameterName))
	{	// write error
		String s = AXIS_ERROR_MSG_MISSING_BLOCK_PARAM;
		s.append(afd::AxisInputLanguage::PartSyntax.ElementDescriptionParameterName);
		GetParseContext().RegisterEvent(asmm::ErrorMessage(0x300515, s));
		hasErrors = true;
	}
	if (!hasErrors)
	{
		String elementTypeId = paramList.GetParameterValue(
      afd::AxisInputLanguage::PartSyntax.ElementTypeParameterName).ToString();
		aslse::ParameterValue& v = paramList.GetParameterValue(
      afd::AxisInputLanguage::PartSyntax.ElementDescriptionParameterName);
		return new aapc::SectionDefinition(elementTypeId, v.Clone());
	}
	else
	{	// fail
		return NULL;
	}
}

template <class ElementLocator, class MaterialLocator>
void aapps::PartParserTemplate<ElementLocator, MaterialLocator>::DoStartContext( void )
{
	// parse element construction metadata
	_sectionDefinition = BuildElementDescription(_sectionParameters);

	// we need to make sure that the section (that is, the element type and its
	// description) is correct and nothing is missing

	// of course, we cannot do that if no description could be build (that is,
	// we are on error recovery condition)
	if (_sectionDefinition != NULL)
	{
		if (!_elementProvider.CanBuildElement(*_sectionDefinition) )
		{	// write error
			String s = _T("Cannot define a part due to: (1) missing parameters; (2) unrecognized or mispelled parameter and/or (3) invalid parameter value.");
			GetParseContext().RegisterEvent(asmm::ErrorMessage(0x30052F, s));
		}
	}
}

template <class ElementLocator, class MaterialLocator>
adc::ElementSet& aapps::PartParserTemplate<ElementLocator, MaterialLocator>::EnsureElementSet( 
  const axis::String& elementSetId )
{
  ada::NumericalModel& model = GetAnalysis().GetNumericalModel();
	if (!model.ExistsElementSet(elementSetId))
	{
    aapc::SymbolTable& st = GetParseContext().Symbols();
		model.AddElementSet(elementSetId, *new adc::ElementSet());
		st.DefineOrRefreshSymbol(elementSetId, aapc::SymbolTable::kElementSet);
	}
	return model.GetElementSet(elementSetId);
}

template <class ElementLocator, class MaterialLocator>
adm::MaterialModel& aapps::PartParserTemplate<ElementLocator, MaterialLocator>::BuildMaterial( 
  const axis::String& materialTypeId, const aslse::ParameterList& materialParams ) const
{
	adm::MaterialModel& materialModel = _materialProvider.BuildMaterial(materialTypeId, materialParams);
	return materialModel;
}
