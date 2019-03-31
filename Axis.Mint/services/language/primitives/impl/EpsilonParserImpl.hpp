#pragma once
#include "../ParserImpl.hpp"

namespace axis { namespace services	{	namespace language { namespace primitives	{	namespace impl {

class EpsilonParserImpl : public axis::services::language::primitives::ParserImpl
{
public:
	EpsilonParserImpl(void);
	~EpsilonParserImpl(void);
private:
	virtual axis::services::language::parsing::ParseResult DoParse(
    const axis::services::language::iterators::InputIterator& begin, 
    const axis::services::language::iterators::InputIterator& end, bool trimSpaces = true) const;
};					
				
}	}	} } } // namespace axis::services::language::primitives::impl
