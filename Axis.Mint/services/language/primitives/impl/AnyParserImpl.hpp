#pragma once
#include "../ParserImpl.hpp"
#include "AxisString.hpp"

namespace axis { namespace services	{ namespace language { namespace primitives { namespace impl {

class AnyParserImpl : public axis::services::language::primitives::ParserImpl
{
public:
  enum ResultType
  {
	  kOperator = 0,
	  kReservedWord = 1,
	  kSpecificId = 2
  };

  AnyParserImpl(const axis::String& expectedValue, const ResultType resultType, 
    int associatedValue = 0, int precedence = 0, int associativity = 0);
  ~AnyParserImpl(void);
private:
  enum CharType
  {
	  kIdentifierDeclarationType = 0,
	  kNonIdentifierDeclarationType = 1
  };

  virtual axis::services::language::parsing::ParseResult DoParse(
    const axis::services::language::iterators::InputIterator& begin, 
    const axis::services::language::iterators::InputIterator& end, bool trimSpaces = true) const;
  CharType CheckCharType(axis::String::value_type c) const;
  void SkipWhitespaces(axis::services::language::iterators::InputIterator& it, 
    const axis::services::language::iterators::InputIterator& end) const;

  const axis::String expectedValue_;
  const ResultType resultType_;  
  const int associatedValue_;
  const int associativity_;
  const int precedence_;
};					

}	} } } } // namespace axis::services::language::primitives::impl

