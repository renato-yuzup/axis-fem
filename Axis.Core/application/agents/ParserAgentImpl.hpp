#pragma once
#include "ParserAgent.hpp"
#include "nocopy.hpp"
#include "services/messaging/CollectorHub.hpp"

namespace axis { namespace application { namespace agents {

class ParserAgent::ParserAgentImpl : public axis::services::messaging::CollectorHub
{
public:
	ParserAgentImpl(axis::services::management::GlobalProviderCatalog& catalog);
	~ParserAgentImpl(void);
	void AddPreProcessorSymbol(const axis::String& symbolName);
	void ClearPreProcessorSymbols(void);
  void BuildAnalysis(const axis::String& outputPath);
	axis::application::jobs::StructuralAnalysis& GetAnalysis(void) const;
	void Parse(const axis::String& masterFileName, 
			   const axis::String& baseIncludePath);
  axis::String GetPreProcessorSymbolList(void) const;
private:
	axis::services::management::GlobalProviderCatalog *catalog_;
	axis::application::jobs::StructuralAnalysis *analysis_;
	axis::application::agents::ParserEngine engine_;
  axis::String symbolList_;

	DISALLOW_COPY_AND_ASSIGN(ParserAgentImpl);
};

} } } // namespace axis::application::agents
