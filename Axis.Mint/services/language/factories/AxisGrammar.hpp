#pragma once
#include "foundation/Axis.Mint.hpp"
#include "../primitives/Parser.hpp"

namespace axis { namespace services { namespace language { namespace factories {

class AXISMINT_API AxisGrammar
{
public:
	static axis::services::language::primitives::Parser CreateEpsilonParser( void );
	static axis::services::language::primitives::Parser CreateIdParser( void );
	static axis::services::language::primitives::Parser CreateNumberParser( void );
	static axis::services::language::primitives::Parser CreateBlankParser( 
    bool enforceSpace = false );
	static axis::services::language::primitives::Parser CreateStringParser( 
    bool acceptEscapedSequence = false);
	static axis::services::language::primitives::Parser CreateOperatorParser( 
    const axis::String& operatorSpelling, int associatedValue = 0, int precedence = 0, 
    int associativity = 0 );
	static axis::services::language::primitives::Parser CreateReservedWordParser( 
    const axis::String& operatorSpelling, int associatedValue = 0 );
	static axis::services::language::primitives::Parser CreateEoiParser(void);

	static axis::services::language::primitives::Parser CreateReservedWordBlockHeadParser(void);
	static axis::services::language::primitives::Parser CreateReservedWordBlockTailParser(void);
	static axis::services::language::primitives::Parser CreateReservedWordParameterListStarterParser(void);

	static bool IsReservedWord(const axis::String& word);

	static bool IsValidToken(const axis::String& symbol);
	static axis::services::language::iterators::InputIterator ExtractNextToken(
    const axis::services::language::iterators::InputIterator& begin, 
    const axis::services::language::iterators::InputIterator& end);
	static axis::services::language::iterators::InputIterator ExtractNextOperator(
    const axis::services::language::iterators::InputIterator& begin, 
    const axis::services::language::iterators::InputIterator& end);
	static axis::services::language::iterators::InputIterator ExtractNextReservedWord(
    const axis::services::language::iterators::InputIterator& begin, 
    const axis::services::language::iterators::InputIterator& end);
private:
	// This class cannot be instantiated (even because there is no reason for it)
	AxisGrammar(void);
};	

} } } } // namespace axis::services::language::factories

