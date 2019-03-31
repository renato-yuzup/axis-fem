#pragma once
#include "foundation/Axis.Mint.hpp"
#include "ParserAction.hpp"

namespace axis { namespace services { namespace language { namespace actions {

class AXISMINT_API ChainedAction : public ParserAction
{
public:
	ChainedAction(const ParserAction& myAction, const ParserAction& nextAction);
	~ChainedAction(void);

	/*
		Execute this action passing the value extracted by the lexer.
	*/
	virtual void Run(const axis::services::language::parsing::ParseResult& result) const;

	/*
		Creates a copy of this object.
	*/
	virtual ParserAction& Clone(void) const;
private:
	ParserAction *_myAction;
	ParserAction *_nextAction;
};		

} } } } // namespace axis::services::language::actions
