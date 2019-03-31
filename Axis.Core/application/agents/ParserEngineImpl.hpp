#pragma once
#include "ParserEngine.hpp"
#include "application/jobs/StructuralAnalysis.hpp"
#include "application/parsing/core/ParseContext.hpp"
#include "application/parsing/core/StatementDecoder.hpp"
#include "application/parsing/preprocessing/InputStack.hpp"
#include "application/parsing/preprocessing/PreProcessor.hpp"
#include "services/management/GlobalProviderCatalog.hpp"
#include "services/messaging/CollectorHub.hpp"
#include "nocopy.hpp"

namespace axis { namespace application { namespace agents {

class ParserEngine::ParserEngineImpl : public axis::services::messaging::CollectorHub
{
public:
  ParserEngineImpl(axis::services::management::GlobalProviderCatalog& manager);
	~ParserEngineImpl(void);
  void AddPreProcessorSymbol(const axis::String& symbolName);
  void ClearPreProcessorSymbols(void);
  void Parse(axis::application::jobs::StructuralAnalysis& analysis, 
    const axis::String& masterInputFilename, const axis::String& baseIncludePath);
private:
  void ResetFileStack(const axis::String& masterInputFilename);
  axis::application::parsing::parsers::BlockParser& GetRootParser(void) const;
  void RunParseRound(axis::application::parsing::core::StatementDecoder& decoder);
  bool DoesNeedNewReadRound(unsigned long lastRoundDefinedReferencesCount, 
                            unsigned long lastRoundUndefinedReferencesCount, 
                            unsigned long roundCount) const;
  void SelectParseRunMode(unsigned long lastRoundDefinedReferencesCount, 
                          unsigned long lastRoundUndefinedReferencesCount) const;

  axis::services::management::GlobalProviderCatalog *catalog_;
  axis::application::jobs::StructuralAnalysis *analysis_;
	axis::application::parsing::preprocessing::InputStack *fileStack_;
	axis::application::parsing::core::ParseContextConcrete *parseContext_;
	axis::application::parsing::preprocessing::PreProcessor *preProcessor_;

  DISALLOW_COPY_AND_ASSIGN(ParserEngineImpl);
};

} } } // namespace axis::application::agents


