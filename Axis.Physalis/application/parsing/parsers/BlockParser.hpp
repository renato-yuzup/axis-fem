#pragma once
#include "foundation/Axis.Physalis.hpp"
#include "AxisString.hpp"
#include "application/jobs/StructuralAnalysis.hpp"
#include "application/parsing/core/ParseContext.hpp"
#include "services/messaging/CollectorEndpoint.hpp"
#include "services/language/syntax/evaluation/ParameterList.hpp"
#include "services/language/iterators/InputIterator.hpp"
#include "services/language/parsing/ParseResult.hpp"

namespace axis { namespace application { namespace parsing { namespace parsers {

class AXISPHYSALIS_API BlockParser : public axis::services::messaging::CollectorEndpoint
{
public:
	BlockParser(void);
	virtual ~BlockParser(void);
	virtual BlockParser& GetNestedContext(const axis::String& contextName, 
    const axis::services::language::syntax::evaluation::ParameterList& paramList) = 0;
	virtual void CloseContext(void);
	virtual void StartContext(axis::application::parsing::core::ParseContext& context);
	virtual axis::services::language::parsing::ParseResult Parse(
    const axis::services::language::iterators::InputIterator& begin, 
    const axis::services::language::iterators::InputIterator& end) = 0;

	/**********************************************************************************************//**
		* <summary> Sets the analysis for which parsing operation is being done.</summary>
		*
		* <param name="a"> [in,out] The analysis.</param>
		**************************************************************************************************/
	void SetAnalysis(axis::application::jobs::StructuralAnalysis& a);

	/**********************************************************************************************//**
		* <summary> Returns the current analysis which we are parsing.</summary>
		*
		* <returns> The current analysis.</returns>
		**************************************************************************************************/
	axis::application::jobs::StructuralAnalysis& GetAnalysis(void) const;

	/**********************************************************************************************//**
		* <summary> Makes this object lose reference to the current analysis so that any subsequent calls
		* 			 to GetAnalysis becomes invalid.</summary>
		**************************************************************************************************/
	void DetachFromAnalysis(void);

	/**********************************************************************************************//**
		* <summary> Returns the current parse context.</summary>
		*
		* <returns> The parse context.</returns>
		**************************************************************************************************/
	axis::application::parsing::core::ParseContext& GetParseContext(void) const;
protected:
  void WarnUnrecognizedParams(
    const axis::services::language::syntax::evaluation::ParameterList& paramList) const;
private:
  virtual void DoCloseContext(void);
  virtual void DoStartContext(void);

  axis::application::jobs::StructuralAnalysis *analysis_;					
  axis::application::parsing::core::ParseContext *parseContext_;
};

} } } } // namespace axis::application::parsing::parsers
