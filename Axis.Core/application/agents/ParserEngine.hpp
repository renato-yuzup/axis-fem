#pragma once

#include "application/jobs/StructuralAnalysis.hpp"
#include "services/management/GlobalProviderCatalog.hpp"
#include "services/messaging/CollectorHub.hpp"
#include "AxisString.hpp"
#include "nocopy.hpp"

namespace axis { namespace application { namespace agents {

/**************************************************************************************************
	* <summary>Translates information contained in the input file into data
	* 			structures in memory usable by the program.</summary>
	**************************************************************************************************/
class ParserEngine : public axis::services::messaging::CollectorHub
{
public:

	/**************************************************************************************************
		* <summary>	Default constructor. </summary>
		**************************************************************************************************/
	ParserEngine(void);

	/**************************************************************************************************
		* <summary>	Destructor. </summary>
		**************************************************************************************************/
	~ParserEngine(void);

	void SetUp(axis::services::management::GlobalProviderCatalog& manager);

	void Parse(	axis::application::jobs::StructuralAnalysis& analysis,
              const axis::String& masterInputFilename,
							const axis::String& baseIncludePath);

	/**************************************************************************************************
		* <summary>	Defines a new preprocessor symbol. </summary>
		*
		* <param name="symbolName">Name of the symbol. </param>
		**************************************************************************************************/
	void AddPreProcessorSymbol(const axis::String& symbolName);

	/**************************************************************************************************
		* <summary>	Clears all defined preprocessor symbols. </summary>
		**************************************************************************************************/
	void ClearPreProcessorSymbols(void);
private:
	class ParserEngineImpl;
	ParserEngineImpl *pimpl_;

  DISALLOW_COPY_AND_ASSIGN(ParserEngine);
};

} } }  // namespace axis::application::agents
