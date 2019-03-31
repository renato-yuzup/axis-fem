#include "NumberParserImpl.hpp"
#include <boost/spirit/include/phoenix_core.hpp>
#include <boost/spirit/include/phoenix_operator.hpp>
#include <boost/spirit/include/phoenix_stl.hpp>
//#include <boost/spirit/home/phoenix/container.hpp>
//#include <boost/spirit/home/phoenix/statement/sequence.hpp>
#include <climits>
#include "../../parsing/EmptyNode.hpp"
#include "../../parsing/NumberTerminal.hpp"
#include "string_traits.hpp"

namespace bs = boost::spirit;
namespace asli = axis::services::language::iterators;
namespace aslp = axis::services::language::parsing;
namespace aslppi = axis::services::language::primitives::impl;

aslppi::NumberParserImpl::NumberParserImpl( void )
{
	InitGrammar();
}

void aslppi::NumberParserImpl::InitGrammar( void )
{
	_signal = 
    bs::qi::lexeme[_T("+")][bs::qi::_val = _T("+")] | bs::qi::lexeme[_T("-")][bs::qi::_val = _T("-")]; 
	_digit_sequence %= +bs::qi::digit;

	_integer = (_signal >> _digit_sequence)[bs::qi::_val = bs::qi::_1 + bs::qi::_2] | 
			   (_digit_sequence)[bs::qi::_val = bs::qi::_1];
	_decimal_part = 
    (_digit_sequence >> bs::qi::lexeme[_T(".")] >> _digit_sequence)[bs::qi::_val = bs::qi::_1 + _T(".") + bs::qi::_2] |
		(_digit_sequence >> bs::qi::lexeme[_T(".")])[bs::qi::_val = bs::qi::_1 + _T(".")] |
		(_digit_sequence)[bs::qi::_val = bs::qi::_1] | 			 
		(bs::qi::lexeme[_T(".")] >> _digit_sequence)[bs::qi::_val = String(_T(".")) + bs::qi::_1];
 	_scientific_not = (bs::qi::lexeme[_T("E")] >> _integer)[bs::qi::_val = String(_T("E")) + bs::qi::_1] |
 		(bs::qi::lexeme[_T("e")] >> _integer)[bs::qi::_val =  String(_T("e")) + bs::qi::_1];
	_num_rule = 
    (_signal >> _decimal_part >> _scientific_not)[bs::qi::_val = bs::qi::_1 + bs::qi::_2 + bs::qi::_3] |
		(_signal >> _decimal_part)[bs::qi::_val = bs::qi::_1 + bs::qi::_2] |
		(_decimal_part >> _scientific_not)[bs::qi::_val = bs::qi::_1 + bs::qi::_2] |
		_decimal_part[bs::qi::_val = bs::qi::_1];
}

aslppi::NumberParserImpl::~NumberParserImpl( void )
{
	/* nothing to do here */
}

aslp::ParseResult aslppi::NumberParserImpl::DoParse( const asli::InputIterator& begin, 
                                                     const asli::InputIterator& end, 
                                                     bool trimSpaces /*= true*/ ) const
{
	bool isInteger;		// flag indicating that we have an integer number 
	size_t intNumSize = 0, doubleNumSize = 0;	// auxiliary variables for measurement purposes

	/* Parse result data */
	aslp::ParseTreeNode *rootNode;
	asli::InputIterator it;
	aslp::ParseResult::Result result;
	String value;
	bool ret;
	
	// check which parse rule better fits in this input stream
	it = begin;
	if(trimSpaces) bs::qi::parse(it, end, *bs::qi::blank); // try to remove whitespaces before, if asked
	ret = bs::qi::parse(it, end, _integer, value);	/* parse as int */
	if (ret) intNumSize = value.size();
	
	it = begin;
	if(trimSpaces) bs::qi::parse(it, end, *bs::qi::blank); // try to remove whitespaces before, if asked
	ret = bs::qi::parse(it, end, _num_rule, value);	/* parse as double */ 
	if (ret) doubleNumSize = value.size();
	
	if (ret)
	{	// since we have a number, parse its value from the string
		isInteger = (intNumSize >= doubleNumSize);	// actually it cannot be greater than, but anyways...

		// remove whitespaces at the end, if asked
		if(trimSpaces) bs::qi::parse(it, end, *bs::qi::blank);	

		axis::String::iterator testIt;
		if (isInteger)
		{	// parse as an integer
			long intVal;
			if(bs::qi::parse(testIt = value.begin(), value.end(), bs::qi::long_, intVal) && 
        testIt == value.end())
			{	
				rootNode = new aslp::NumberTerminal(intVal, value);
				result = aslp::ParseResult::MatchOk;
			}
			else
			{	// indeed it is an integer, but even so we couldn't parse; this mean we detected an overflow!
				result = aslp::ParseResult::FailedMatch;
				it = begin;		// move iterator to the beginning
				rootNode = new aslp::EmptyNode();
			}
		}
		else
		{	// parse as a double
			double doubleVal;
			if (bs::qi::parse(testIt = value.begin(), value.end(), bs::qi::double_, doubleVal) 
        && testIt == value.end() && /* must parse successfully and ...*/
				doubleVal != std::numeric_limits<double>::infinity() &&	 /* should not be equal to positive infinity and ...*/
				doubleVal != -std::numeric_limits<double>::infinity() && /* neither negative infinity ... */
				doubleVal != std::numeric_limits<double>::quiet_NaN()		 /* nor a NaN (not a number) */
				)	
			{
				rootNode = new aslp::NumberTerminal(doubleVal, value);
				result = aslp::ParseResult::MatchOk;
			}
			else
			{	// indeed it is a double, but even so we couldn't parse (or one of the other conditions was 
				// not met -- see above); this mean we detected an overflow/underflow!
				result = aslp::ParseResult::FailedMatch;
				it = begin;		// move iterator to the beginning
				rootNode = new aslp::EmptyNode();
			}
		}
	}
	else
	{	// this is not a number!
		rootNode = new aslp::EmptyNode();
		result = aslp::ParseResult::FailedMatch;
	}
	return aslp::ParseResult(result, *rootNode, it);
}
