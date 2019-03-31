#pragma once
#include "application/parsing/parsers/BlockParser.hpp"

namespace axis
{
namespace services { namespace language { namespace syntax { namespace evaluation {
class ParameterList;
} } } }

namespace application
{
namespace output {
class ResultBucketConcrete;
}
namespace factories { namespace parsers {
class StepParserProvider;
} }

namespace locators
{     
class SolverFactoryLocator;
class ClockworkFactoryLocator;
class CollectorFactoryLocator;
class WorkbookFactoryLocator;
}
namespace parsing
{
namespace parsers
{

// Describing this class with template arguments in order to support mocking
template <class SolverFactoryLoc, class ClockworkFactoryLoc>
class StepParserTemplate : public axis::application::parsing::parsers::BlockParser
{
public:
	StepParserTemplate(
        axis::application::factories::parsers::StepParserProvider& parentProvider, 
        const axis::String& stepName,
				SolverFactoryLoc& solverLocator,
				const axis::String& solverType, 
				real startTime, real endTime, 
				const axis::services::language::syntax::evaluation::ParameterList& solverParams,
        axis::application::locators::CollectorFactoryLocator& collectorLocator,
        axis::application::locators::WorkbookFactoryLocator& formatLocator);
	StepParserTemplate(
        axis::application::factories::parsers::StepParserProvider& parentProvider, 
        const axis::String& stepName,
				SolverFactoryLoc& solverLocator,
				ClockworkFactoryLoc& clockworkLocator,
				const axis::String& solverType, 
				real startTime, real endTime, 
				const axis::services::language::syntax::evaluation::ParameterList& solverParams,
				const axis::String& clockworkType, 
				const axis::services::language::syntax::evaluation::ParameterList& clockworkParams,
        axis::application::locators::CollectorFactoryLocator& collectorLocator,
        axis::application::locators::WorkbookFactoryLocator& formatLocator);
	virtual ~StepParserTemplate(void);
	virtual BlockParser& GetNestedContext( 
      const axis::String& contextName, 
      const axis::services::language::syntax::evaluation::ParameterList& paramList );
	virtual axis::services::language::parsing::ParseResult Parse( 
      const axis::services::language::iterators::InputIterator& begin, 
      const axis::services::language::iterators::InputIterator& end );
	virtual void DoStartContext( void );
private:
  void Init(const axis::String& solverType, 
        real startTime, real endTime, 
        const axis::services::language::syntax::evaluation::ParameterList& solverParams);
  bool ValidateCollectorBlockInformation(
        const axis::services::language::syntax::evaluation::ParameterList& contextParams);
  axis::application::parsing::parsers::BlockParser& CreateResultCollectorParser(
        const axis::services::language::syntax::evaluation::ParameterList& contextParams) const;

  axis::application::factories::parsers::StepParserProvider& provider_;
  SolverFactoryLoc& solverLocator_;
  ClockworkFactoryLoc *clockworkLocator_;
  axis::application::locators::CollectorFactoryLocator& collectorLocator_;
  axis::application::locators::WorkbookFactoryLocator& formatLocator_;
  axis::String stepName_;
  real stepStartTime_, stepEndTime_;
  axis::String solverTypeName_;
  axis::services::language::syntax::evaluation::ParameterList *solverParams_;
  axis::String clockworkTypeName_;
  axis::services::language::syntax::evaluation::ParameterList *clockworkParams_;
  bool isClockworkDeclared_;				
  // used to determine the read state of the input files
  bool isNewReadRound_;
  bool dirtyStepBlock_;
  axis::application::parsing::parsers::BlockParser *nullParser_;
  axis::application::output::ResultBucketConcrete *stepResultBucket_;
};		

typedef StepParserTemplate<axis::application::locators::SolverFactoryLocator, 
  axis::application::locators::ClockworkFactoryLocator> StepParser;

} } } } // namespace axis::application::parsing::parsers
