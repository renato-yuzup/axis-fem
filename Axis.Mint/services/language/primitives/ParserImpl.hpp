#pragma once
#include "../parsing/ParseResult.hpp"
#include "../actions/ParserAction.hpp"

namespace axis { namespace services	{	namespace language { namespace primitives	{

class ParserImpl
{
public:
	ParserImpl(void);
	virtual ~ParserImpl(void);
	void NotifyDestroy(void);
	void NotifyUse(void);
	
  inline axis::services::language::parsing::ParseResult Parse(
    const axis::services::language::iterators::InputIterator& begin, 
    const axis::services::language::iterators::InputIterator& end, bool trimSpaces = true) const
  {
    axis::services::language::parsing::ParseResult result = DoParse(begin, end, trimSpaces);
    if (result.IsMatch()) action_->Run(result);
    return result;
  }

	void AddAction(const actions::ParserAction& action);
	ParserImpl& operator <<(const axis::services::language::actions::ParserAction& action);
private:
  virtual axis::services::language::parsing::ParseResult DoParse(
    const axis::services::language::iterators::InputIterator& begin, 
    const axis::services::language::iterators::InputIterator& end, bool trimSpaces = true) const = 0;

	int refCount_;
	axis::services::language::actions::ParserAction *action_;
};			

}	} } } // namespace axis::services::language::primitives
