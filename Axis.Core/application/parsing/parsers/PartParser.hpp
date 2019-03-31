#ifndef __PARTPARSER_HPP
#define __PARTPARSER_HPP

#include "application/parsing/parsers/BlockParser.hpp"
#include "application/factories/parsers/BlockProvider.hpp"
#include "application/locators/ElementParserLocator.hpp"
#include "application/locators/MaterialFactoryLocator.hpp"
#include "application/parsing/core/SectionDefinition.hpp"
#include "domain/collections/ElementSet.hpp"
#include "domain/materials/MaterialModel.hpp"
#include "services/language/primitives/OrExpressionParser.hpp"
#include "services/language/primitives/AssignmentParser.hpp"
#include "services/language/primitives/EnumerationParser.hpp"
#include "services/language/syntax/evaluation/ParameterList.hpp"
#include "application/parsing/core/SectionDefinition.hpp"


namespace axis { namespace application { namespace parsing { namespace parsers {

template <class ElementLocator, class MaterialLocator>
class PartParserTemplate : public axis::application::parsing::parsers::BlockParser
{
public:
	PartParserTemplate(axis::application::factories::parsers::BlockProvider& parent, 
    ElementLocator& elementProvider, MaterialLocator& constitutiveModelProvider, 
    const axis::services::language::syntax::evaluation::ParameterList& paramList);
	virtual ~PartParserTemplate(void);
	virtual axis::application::parsing::parsers::BlockParser& GetNestedContext(
    const axis::String& contextName, 
    const axis::services::language::syntax::evaluation::ParameterList& paramList );
	virtual axis::services::language::parsing::ParseResult Parse(
    const axis::services::language::iterators::InputIterator& begin, 
    const axis::services::language::iterators::InputIterator& end);
protected:
	virtual void DoStartContext(void);
private:
  void InitGrammar(void);
  axis::domain::materials::MaterialModel& BuildMaterial(const axis::String& materialTypeId, 
    const axis::services::language::syntax::evaluation::ParameterList& materialParams) const;
  axis::application::parsing::core::SectionDefinition * BuildElementDescription(
    const axis::services::language::syntax::evaluation::ParameterList& paramList) const;
  axis::domain::collections::ElementSet& EnsureElementSet(const axis::String& elementSetId );

  MaterialLocator& _materialProvider;
  ElementLocator& _elementProvider;
  axis::application::factories::parsers::BlockProvider& _parent;
  axis::services::language::syntax::evaluation::ParameterList& _sectionParameters;

  // our grammar rules	
  axis::services::language::primitives::OrExpressionParser      _enumParam;
  axis::services::language::primitives::EnumerationParser       _paramList;
  axis::services::language::primitives::AssignmentParser        _assign;
  axis::services::language::primitives::OrExpressionParser      _value;
  axis::services::language::primitives::GeneralExpressionParser _array;
  axis::services::language::primitives::OrExpressionParser      _arrayContents;
  axis::services::language::primitives::EnumerationParser       _arrayEnum;
  axis::services::language::primitives::OrExpressionParser      _arrayVal;
  axis::services::language::primitives::OrExpressionParser      _elementSetId;
  axis::services::language::primitives::GeneralExpressionParser _materialExpression;

  // this variable holds element creation parameters
  axis::application::parsing::core::SectionDefinition *_sectionDefinition;
};				

typedef PartParserTemplate<axis::application::locators::ElementParserLocator,
  axis::application::locators::MaterialFactoryLocator> PartParser;

} } } } // namespace axis::application::parsing::parsers

#endif