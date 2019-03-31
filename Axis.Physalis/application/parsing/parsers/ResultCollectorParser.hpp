#pragma once
#include "application/parsing/parsers/BlockParser.hpp"

namespace axis
{
namespace services { namespace language { 
namespace iterators {
class InputIterator;
}  
namespace syntax { namespace evaluation {
class ParameterList;
} }
} }

namespace application { 
  
namespace locators {
class WorkbookFactoryLocator;
class CollectorFactoryLocator;
}

namespace output { 
class ResultBucketConcrete;
class ResultDatabase;

namespace collectors {
class ElementSetCollector;
class GenericCollector;
class NodeSetCollector;
} }

namespace parsing { namespace parsers {

class ResultCollectorParser : public axis::application::parsing::parsers::BlockParser
{
public:
	ResultCollectorParser( axis::application::locators::WorkbookFactoryLocator& formatLocator,
    axis::application::locators::CollectorFactoryLocator& collectorLocator, 
    axis::application::output::ResultBucketConcrete& resultBucket,
    const axis::String& fileName, const axis::String& formatName, 
    const axis::services::language::syntax::evaluation::ParameterList& formatArguments,
    bool append);
	virtual ~ResultCollectorParser(void);

	virtual BlockParser& GetNestedContext( 
    const axis::String& contextName, 
    const axis::services::language::syntax::evaluation::ParameterList& paramList );

	virtual axis::services::language::parsing::ParseResult Parse( 
    const axis::services::language::iterators::InputIterator& begin, 
    const axis::services::language::iterators::InputIterator& end );
protected:
  virtual void DoCloseContext( void );
  virtual void DoStartContext( void );
private:
  bool ShouldDiscardDatabase(void);
  void DestroyDatabase(void);
  void MarkFileNameAsUsed(const axis::String& formatExtension);
  axis::String GetChainGlobalSymbolName(const axis::String& fileName) const;
  axis::String GetChainPerStepSymbolName(const axis::String& fileName, int stepIndex) const;
  axis::String GetAddedChainGlobalSymbolName(const axis::String& fileName) const;
  axis::String GetSymbolName(const axis::String& fileName, const axis::String& formatName) const;
  bool TryAcquireOutputStream( const axis::String& filename );

  axis::application::locators::CollectorFactoryLocator& collectorLocator_;
  axis::application::locators::WorkbookFactoryLocator& workbookLocator_;
  axis::services::language::syntax::evaluation::ParameterList& formatArguments_;
  axis::application::output::ResultBucketConcrete& resultBucket_;
  axis::application::output::ResultDatabase *currentDatabase_;

  axis::String fileName_;    // filename for the format file
  axis::String formatName_;  // name of the workbook format
  bool onErrorRecovering_;   // a parser error occurred; skip parsing block inner statements
  bool onSkipBuild_;         // an event triggered skipping collector construction
  bool appendToFile_;        // user requested to open format file in append mode
};

} } } } // namespace axis::application::parsing::parsers
