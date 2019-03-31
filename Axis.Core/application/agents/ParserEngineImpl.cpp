#include "ParserEngineImpl.hpp"
#include "application/factories/parsers/BlockProvider.hpp"
#include "application/parsing/parsers/BlockParser.hpp"
#include "services/management/ServiceLocator.hpp"
#include "services/language/syntax/evaluation/ParameterList.hpp"
#include "application/parsing/core/EventStatistic.hpp"
#include "application/parsing/core/StatementDecoder.hpp"
#include "application/parsing/core/SymbolTable.hpp"
#include "services/messaging/ErrorMessage.hpp"
#include "application/parsing/error_messages.hpp"
#include "messages/ParseRoundEndMessage.hpp"
#include "messages/ParseFinishMessage.hpp"

namespace aaag = axis::application::agents;
namespace aaj = axis::application::jobs;
namespace aapc = axis::application::parsing::core;
namespace aappar = axis::application::parsing::parsers;
namespace aappre = axis::application::parsing::preprocessing;
namespace asmm = axis::services::messaging;
namespace aafp = axis::application::factories::parsers;
namespace aapps = axis::application::parsing::parsers;
namespace asmg = axis::services::management;
namespace aslse = axis::services::language::syntax::evaluation;
namespace aaam = axis::application::agents::messages;

static const int kMaxReadRounds = 9;

aaag::ParserEngine::ParserEngineImpl::ParserEngineImpl( asmg::GlobalProviderCatalog& manager)
{
  catalog_      = &manager;
  fileStack_    = new aappre::InputStack();
  parseContext_ = new aapc::ParseContextConcrete();
  preProcessor_ = new aappre::PreProcessor(*fileStack_, *parseContext_);
}

aaag::ParserEngine::ParserEngineImpl::~ParserEngineImpl( void )
{
  delete preProcessor_;
  delete parseContext_;
  delete fileStack_;
}

void aaag::ParserEngine::ParserEngineImpl::AddPreProcessorSymbol( const axis::String& symbolName )
{
  preProcessor_->AddPreProcessorSymbol(symbolName);
}

void aaag::ParserEngine::ParserEngineImpl::ClearPreProcessorSymbols( void )
{
  preProcessor_->ClearPreProcessorSymbols();
}

void aaag::ParserEngine::ParserEngineImpl::Parse( aaj::StructuralAnalysis& analysis, 
                                                  const axis::String& masterInputFilename, 
                                                  const axis::String& baseIncludePath )
{
  // used to determine if we must do a new read round
  long lastRoundUndefinedReferencesCount = 0;
  long lastRoundDefinedReferencesCount = 0;
  int roundCount = 0;
  bool needsNewRound;
  analysis_ = &analysis;
  preProcessor_->SetBaseIncludePath(baseIncludePath);
  parseContext_->ConnectListener(*this);
  try
  {
    do	// main multi-pass loop
    {
      needsNewRound = false;  // let's suppose we won't need a new read round
      ResetFileStack(masterInputFilename);
      preProcessor_->Reset();
      preProcessor_->Prepare();
      aapps::BlockParser& analysisParser = GetRootParser();
      analysisParser.SetAnalysis(*analysis_);
      aapc::StatementDecoder decoder(*parseContext_);
      decoder.SetCurrentContext(analysisParser);

      RunParseRound(decoder); // start parse round!
      roundCount++;
      needsNewRound = DoesNeedNewReadRound(lastRoundDefinedReferencesCount, lastRoundUndefinedReferencesCount, roundCount);

      // check for extreme situations
      if (roundCount == kMaxReadRounds && needsNewRound)
      {	// too much read rounds and, still, we couldn't resolve all references -- this a symptom
        // of unexpected parser behavior
        String s = AXIS_ERROR_MSG_INFINITE_READ;
        s.replace(_T("%1"), String::int_parse((long)roundCount));
        parseContext_->RegisterEvent(asmm::ErrorMessage(0x300401, s, _T("Load analysis error")));
        needsNewRound = false;
      }
      // abort operation if too many errors were found
      if (parseContext_->EventSummary().GetErrorCount() >= aapc::ParseContext::MaxAllowableErrorCount)
      {	// too many errors -- we can't continue
        parseContext_->RegisterEvent(
          asmm::ErrorMessage(0x300402, AXIS_ERROR_MSG_TOO_MANY_ERRORS, _T("Load analysis error")));
        needsNewRound = false;	// force abort
      }

      if (needsNewRound)
      {
        aapc::SymbolTable& st = parseContext_->Symbols();
        SelectParseRunMode(lastRoundDefinedReferencesCount, lastRoundUndefinedReferencesCount);
        // store last read summary
        lastRoundUndefinedReferencesCount = st.GetRoundUnresolvedReferenceCount();
        lastRoundDefinedReferencesCount   = st.GetRoundDefinedReferenceCount();
        parseContext_->AdvanceRound();

        DispatchMessage(aaam::ParseRoundEndMessage(*analysis_, 
                                                   parseContext_->EventSummary().GetErrorCount(), 
                                                   parseContext_->EventSummary().GetWarningCount(), 
                                                   roundCount,
                                                   lastRoundDefinedReferencesCount, 
                                                   lastRoundUndefinedReferencesCount));
      }
    } while (needsNewRound);
  }
  catch (axis::foundation::AxisException& ex)
  {
    // report error
    unsigned long lineIdx = preProcessor_->GetLastLineReadIndex();
    String sourceFileName = preProcessor_->GetLastLineSourceName();
    asmm::ErrorMessage msg(0x3004FF, String(
      _T("A critical error occurred when parsing model file. Read round = %1.\n   Source: %2 (line %3)"))
      .replace(_T("%1"), String::int_parse((long)roundCount))
      .replace(_T("%2"), sourceFileName)
      .replace(_T("%3"), String::int_parse(lineIdx)), 
      _T("Load analysis error"), ex);
    parseContext_->RegisterEvent(msg);
  }
  catch (...)
  {
    // report error
    unsigned long lineIdx = preProcessor_->GetLastLineReadIndex();
    String sourceFileName = preProcessor_->GetLastLineSourceName();
    asmm::ErrorMessage msg(0x3004FF, String(
      _T("A critical error occurred when parsing model file. Read round = %1.\n   Source: %2 (line %3)"))
      .replace(_T("%1"), String::int_parse((long)roundCount))
      .replace(_T("%2"), sourceFileName)
      .replace(_T("%3"), String::int_parse(lineIdx)), 
      _T("Load analysis error"));
    parseContext_->RegisterEvent(msg);
  }

  parseContext_->DisconnectListener(*this);
  DispatchMessage(aaam::ParseFinishMessage(*analysis_, 
                                           parseContext_->EventSummary().GetErrorCount(), 
                                           parseContext_->EventSummary().GetWarningCount(), 
                                           roundCount));
  analysis_ = NULL;
}

void aaag::ParserEngine::ParserEngineImpl::ResetFileStack( const axis::String& masterInputFilename )
{
  // remove included files from the stack
  while (fileStack_->Count() > 1)
  {
    fileStack_->CloseTopStream();
  }

  if (fileStack_->Count() == 1)
  {
    // rewind file pointer to beginning
    fileStack_->GetTopStream().Reset();
  }
  else
  {
    fileStack_->AddStream(masterInputFilename);
  }
}

aapps::BlockParser& aaag::ParserEngine::ParserEngineImpl::GetRootParser( void ) const
{
  aafp::BlockProvider& analysisProvider = (aafp::BlockProvider&)
    catalog_->GetProvider(asmg::ServiceLocator::GetMasterInputParserProviderPath());
  aapps::BlockParser& analysisParser = 
    analysisProvider.BuildParser(_T(""), aslse::ParameterList::Empty);
  return analysisParser;
}

void aaag::ParserEngine::ParserEngineImpl::RunParseRound( aapc::StatementDecoder& decoder )
{
  long lastBufferSize = 0, currentBufferSize = -1;
  bool shouldContinue = true;
  String inputBuffer;
  String line;
  while (!preProcessor_->IsEOF() && shouldContinue)	// parse while we have contents and no abort request has been made
  {
    line = preProcessor_->ReadLine();	// pop line from preprocessor
    decoder.FeedDecoderBuffer(line);
    currentBufferSize = decoder.GetBufferSize();
    try
    {
      do	// process this line until no statements are left or only partial statements are found
      {
        lastBufferSize = currentBufferSize;
        shouldContinue = decoder.ProcessLine();
        currentBufferSize = decoder.GetBufferSize();
      } while (shouldContinue &&          // that is, no errors occurred and no emergency stop was requested
               currentBufferSize > 0 &&   // there is still more in the buffer to parse
               lastBufferSize != currentBufferSize);  // buffer was processed since the last call
    }
    catch (axis::foundation::AxisException& ex)
    {
      parseContext_->RegisterEvent(asmm::ErrorMessage(0x3004FD, 
        _T("An unforeseen behavior of a parser unit triggered an exception and forced termination of the parsing process."),
        _T("Load analysis error"),
        ex));
      shouldContinue = false;
    }
    catch (...)		// an unknown exception
    {
      parseContext_->RegisterEvent(asmm::ErrorMessage(0x3004FE, 
        _T("An unforeseen behavior of a parser unit triggered an unknown error and forced termination of the parsing process."),
        _T("Load analysis error")));
      shouldContinue = false;
    }
  }
  decoder.EndProcessing();
}

bool aaag::ParserEngine::ParserEngineImpl::DoesNeedNewReadRound( 
  unsigned long lastRoundDefinedReferencesCount, unsigned long lastRoundUndefinedReferencesCount, 
  unsigned long roundCount ) const
{
  aapc::SymbolTable& st = parseContext_->Symbols();
  unsigned long undefinedReferencesCount = st.GetRoundUnresolvedReferenceCount();
  unsigned long definedReferencesCount   = st.GetRoundDefinedReferenceCount();
  // abort operation if too many errors were found
  if (parseContext_->EventSummary().GetErrorCount() >= aapc::ParseContext::MaxAllowableErrorCount)
  {	// too many errors -- we can't continue
    return false;
  }
  if (undefinedReferencesCount > 0)	// we only need a new round if there are any undefined references
  {
    if (parseContext_->GetRunMode() == aapc::ParseContext::kTrialMode)
    {	// some references could not be resolved on first try; we need a collate round
      return true;
    }
    else if (parseContext_->GetRunMode() == aapc::ParseContext::kCollateMode &&
            (lastRoundUndefinedReferencesCount != undefinedReferencesCount || 
            (lastRoundUndefinedReferencesCount == undefinedReferencesCount && 
            lastRoundDefinedReferencesCount != definedReferencesCount)))
    {	// after a collate round, there is still some references missing, although others were resolved
      return true;
    }
    else if (parseContext_->GetRunMode() == aapc::ParseContext::kCollateMode &&
            (lastRoundUndefinedReferencesCount == undefinedReferencesCount && 
            lastRoundDefinedReferencesCount == definedReferencesCount))
    {	// we did a new round and nothing has changed; do a last round to trace back errors
      return true;
    }
  }
  return false;
}


void aaag::ParserEngine::ParserEngineImpl::SelectParseRunMode( 
  unsigned long lastRoundDefinedReferencesCount, unsigned long lastRoundUndefinedReferencesCount ) const
{
  aapc::SymbolTable& st = parseContext_->Symbols();
  unsigned long undefinedReferencesCount = st.GetRoundUnresolvedReferenceCount();
  unsigned long definedReferencesCount   = st.GetRoundDefinedReferenceCount();

  // these are the only situations in which run mode is changed
  if (undefinedReferencesCount > 0)
  {
    if (parseContext_->GetRunMode() == aapc::ParseContext::kTrialMode)
    {	// some references could not be resolved on first try; we need a collate round
      parseContext_->SetRunMode(aapc::ParseContext::kCollateMode);
    }
    else if (parseContext_->GetRunMode() == aapc::ParseContext::kCollateMode &&
      (lastRoundUndefinedReferencesCount == undefinedReferencesCount && 
      lastRoundDefinedReferencesCount == definedReferencesCount))
    {	// we did a new round and nothing has changed; do a last round to trace back errors
      parseContext_->SetRunMode(aapc::ParseContext::kInspectionMode);
    }
  }
}
