#pragma once
#include "foundation/Axis.Mint.hpp"
#include "../parsing/ParseResult.hpp"
#include "../actions/ParserAction.hpp"

namespace axis { namespace services { namespace language { 

namespace factories {
  class AxisGrammar;
} // namespace factories

namespace primitives {

class ParserImpl;

class AXISMINT_API Parser
{
public:
  Parser(const Parser& other);
  ~Parser(void);

  axis::services::language::parsing::ParseResult Parse(
    const axis::services::language::iterators::InputIterator& begin, 
    const axis::services::language::iterators::InputIterator& end, bool trimSpaces = true) const;
  axis::services::language::parsing::ParseResult operator()(
    const axis::services::language::iterators::InputIterator& begin, 
    const axis::services::language::iterators::InputIterator& end, bool trimSpaces = true) const;

  Parser& AddAction(const actions::ParserAction& action);
  Parser& operator <<(const actions::ParserAction& action);

  Parser& operator =(const Parser& other);

  friend class axis::services::language::factories::AxisGrammar;
private:
  void Copy(const Parser& other);
  Parser(ParserImpl *firstParserInChain, ParserImpl *lastParserInChain);
  Parser(ParserImpl& implementationLogic);

	ParserImpl *parserImpl_;
	/* Weak reference to the last parser in the chain we represent */
	ParserImpl *lastParserImplInChain_;
};					

}	// namespace primitives

} } } // namespace axis::services::language

