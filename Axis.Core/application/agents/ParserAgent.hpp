#pragma once
#include "application/agents/ParserEngine.hpp"
#include "application/jobs/StructuralAnalysis.hpp"
#include "application/parsing/core/ParseContext.hpp"
#include "application/parsing/core/StatementDecoder.hpp"
#include "application/parsing/preprocessing/PreProcessor.hpp"
#include "services/management/GlobalProviderCatalog.hpp"
#include "services/messaging/CollectorHub.hpp"
#include "AxisString.hpp"
#include "nocopy.hpp"

namespace axis { namespace application { namespace agents {

/**********************************************************************************************//**
	* <summary> Executes parsing operation on user input files.</summary>
	*
	* <seealso cref="axis::services::messaging::CollectorHub"/>
	**************************************************************************************************/
class ParserAgent : public axis::services::messaging::CollectorHub
{
public:
	/**********************************************************************************************//**
		* <summary> Default constructor.</summary>
		**************************************************************************************************/
	ParserAgent(void);

  ~ParserAgent(void);

	/**********************************************************************************************//**
		* <summary> Adds a preprocessor symbol definition.</summary>
		*
		* <param name="symbolName"> Name of the symbol.</param>
		**************************************************************************************************/
	void AddPreProcessorSymbol(const axis::String& symbolName);

	/**********************************************************************************************//**
		* <summary> Clears definitions of preprocessor symbols.</summary>
		**************************************************************************************************/
	void ClearPreProcessorSymbols(void);

	/**************************************************************************************************
		* <summary>	Sets up this agent. </summary>
		*
		* <param name="manager">	[in,out] The program module manager to use. </param>
		**************************************************************************************************/
	void SetUp(axis::services::management::GlobalProviderCatalog& manager);


	/**********************************************************************************************//**
		* <summary> Reads an analysis from the specified input file.</summary>
		*
		* <param name="masterFileName">	 Filename of the file where parsing will begin.</param>
		* <param name="baseIncludePath">    Full pathname to use when relative paths need to be 
		* 									 resolved on input file search.</param>
		* <param name="outputLocationPath"> Base pathname of output
		* 									 files.</param>
		**************************************************************************************************/
	void ReadAnalysis(const axis::String& masterFileName, 
						const axis::String& baseIncludePath, 
						const axis::String& outputLocationPath);

	/**********************************************************************************************//**
		* <summary> Returns the last read analysis.</summary>
		*
		* <returns> The analysis.</returns>
		**************************************************************************************************/
	axis::application::jobs::StructuralAnalysis& GetAnalysis(void) const;
private:
	class ParserAgentImpl;
  class ParserAgentDispatcher;

  friend class ParserAgentDispatcher;
  virtual void AddTracingInformation( axis::services::messaging::Message& message ) const;
  DISALLOW_COPY_AND_ASSIGN(ParserAgent);

  ParserAgentImpl *pimpl_;
  ParserAgentDispatcher *dispatcher_;
};
} } }

