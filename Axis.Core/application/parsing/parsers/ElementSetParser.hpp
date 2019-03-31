#pragma once
#include "application/factories/parsers/BlockProvider.hpp"
#include "application/parsing/parsers/BlockParser.hpp"
#include "services/language/primitives/GeneralExpressionParser.hpp"
#include "services/language/primitives/AtomicExpressionParser.hpp"
#include "domain/collections/ElementSet.hpp"
#include "foundation/Axis.SystemBase.hpp"

namespace axis { namespace application { namespace parsing { namespace parsers {

class ElementSetParser : public axis::application::parsing::parsers::BlockParser
{
public:
	ElementSetParser(axis::application::factories::parsers::BlockProvider& factory, 
    const axis::services::language::syntax::evaluation::ParameterList& paramList);
	virtual ~ElementSetParser(void);
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
  void AddToElementSet(id_type from, id_type to);

  axis::application::factories::parsers::BlockProvider& _parentProvider;
  axis::services::language::syntax::evaluation::ParameterList& _paramList;
  bool _mustIgnoreParse;
  bool _isFirstTimeRead;
  bool _hasUnresolvedElements;
  axis::domain::collections::ElementSet  *_elementSet;
  axis::String _elementSetAlias;
  axis::services::language::primitives::GeneralExpressionParser _elementRange;
  axis::services::language::primitives::AtomicExpressionParser _elementIdentifier;
};

} } } } // namespace axis::application::parsing::parsers
