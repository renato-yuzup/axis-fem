#include "SkipperParser.hpp"
#include "services/language/factories/AxisGrammar.hpp"

namespace asli = axis::services::language::iterators;
namespace aslf = axis::services::language::factories;
namespace aslp = axis::services::language::parsing;
namespace aslpp = axis::services::language::primitives;
namespace asls = axis::services::language::syntax;

asli::InputIterator asls::SkipperParser::SkipSymbol(const asli::InputIterator& begin, 
                                                    const asli::InputIterator& end) const
{
  asli::InputIterator it = begin;
  asli::InputIterator lastPos;

  aslpp::Parser blankSkipper = aslf::AxisGrammar::CreateBlankParser();
  aslpp::Parser blank        = aslf::AxisGrammar::CreateBlankParser(true);
  aslpp::Parser id           = aslf::AxisGrammar::CreateIdParser();
  aslpp::Parser number       = aslf::AxisGrammar::CreateNumberParser();
  aslpp::Parser string       = aslf::AxisGrammar::CreateStringParser(true);
	aslp::ParseResult result;

	result = id.Parse(begin, end);
	if (result.IsMatch())
	{
		return result.GetLastReadPosition();
	}

	result = number.Parse(begin, end);
	if (result.IsMatch())
	{
		return result.GetLastReadPosition();
	}

	result = string.Parse(begin, end);
	if (result.IsMatch())
	{
		return result.GetLastReadPosition();
	}

	// no matches -- it is an operator, probably
	while (it != end)
	{
		result = id.Parse(it, end);
		if (result.IsMatch()) return blankSkipper(it, end).GetLastReadPosition();
		result = number.Parse(it, end);
		if (result.IsMatch()) return blankSkipper(it, end).GetLastReadPosition();
		result = string.Parse(it, end);
		if (result.IsMatch()) return blankSkipper(it, end).GetLastReadPosition();
		result = blank.Parse(it, end);
		if (result.IsMatch()) return it;

		++it;
	}

	// no symbol found
	return end;
}

asli::InputIterator asls::SkipperParser::operator ()(const asli::InputIterator& begin, 
                                                     const asli::InputIterator& end) const
{
	return SkipSymbol(begin, end);
}
