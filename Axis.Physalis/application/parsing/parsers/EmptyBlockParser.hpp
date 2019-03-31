#pragma once
#include "foundation/Axis.Physalis.hpp"
#include "application/parsing/parsers/BlockParser.hpp"
#include "application/factories/parsers/BlockProvider.hpp"

namespace axis { namespace application { namespace parsing { namespace parsers {

/**********************************************************************************************//**
	* @class	EmptyBlockParser
	*
	* @brief	An empty block parser, which throws an error whenever a non-block declaration 
	* 			is found in its context.
	*
	* @author	Renato T. Yamassaki
	* @date	11 abr 2012
	**************************************************************************************************/
class AXISPHYSALIS_API EmptyBlockParser : public axis::application::parsing::parsers::BlockParser
{
public:
	EmptyBlockParser(axis::application::factories::parsers::BlockProvider& factory);
	virtual ~EmptyBlockParser(void);
	virtual axis::application::parsing::parsers::BlockParser& GetNestedContext(
    const axis::String& contextName, 
    const axis::services::language::syntax::evaluation::ParameterList& paramList);
	virtual axis::services::language::parsing::ParseResult Parse(
    const axis::services::language::iterators::InputIterator& begin, 
    const axis::services::language::iterators::InputIterator& end);
	virtual void DoStartContext( void );
private:
  axis::application::factories::parsers::BlockProvider& _parent;
};		

} } } } // namespace axis::application::parsing::parsers
