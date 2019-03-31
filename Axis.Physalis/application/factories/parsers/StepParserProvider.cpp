#include "StepParserProvider.hpp"
#include "AxisString.hpp"
#include "application/locators/SolverFactoryLocator.hpp"
#include "application/locators/ClockworkFactoryLocator.hpp"
#include "application/locators/WorkbookFactoryLocator.hpp"
#include "application/locators/CollectorFactoryLocator.hpp"
#include "application/parsing/parsers/StepParser.hpp"
#include "services/language/syntax/evaluation/AtomicValue.hpp"
#include "services/language/syntax/evaluation/NumberValue.hpp"
#include "services/language/syntax/evaluation/ParameterList.hpp"
#include "services/management/ServiceLocator.hpp"
#include "foundation/definitions/AxisInputLanguage.hpp"
#include "foundation/InvalidOperationException.hpp"

namespace aafp = axis::application::factories::parsers;
namespace aal = axis::application::locators;
namespace aapps = axis::application::parsing::parsers;
namespace aslse = axis::services::language::syntax::evaluation;
namespace asmg = axis::services::management;
namespace af = axis::foundation;
namespace afdf = axis::foundation::definitions;

aafp::StepParserProvider::StepParserProvider( void )
{
	// nothing to do here
}

aafp::StepParserProvider::~StepParserProvider( void )
{
	// nothing to do here
}

bool aafp::StepParserProvider::CanParse( const axis::String& blockName, const aslse::ParameterList& paramList )
{
	if (blockName != afdf::AxisInputLanguage::StepSyntax.BlockName)
	{
		return false;
	}

	// check for required parameters
	if (!IsRequiredParametersPresentAndValid(paramList))
	{
		return false;
	}
	// ok, even though we didn't check if we have a factory capable
	// to build the specified solver, we will delegate this task to
	// the step parser, so we assume everything is ok for now
	return true;
}

aapps::BlockParser& aafp::StepParserProvider::BuildParser( const axis::String& contextName, 
                                                           const aslse::ParameterList& paramList )
{
	axis::String clockworkTypeName;
  axis::String stepName;
	aslse::ParameterList *clockworkParams = NULL;
	bool isClockworkDeclared = false;

	if (contextName != afdf::AxisInputLanguage::StepSyntax.BlockName)
	{	// huh, did the user check for the parser in the valid provider?
		throw af::InvalidOperationException(_T("Cannot build parser for this context."));
	}

	// check for required parameters
	if (!IsRequiredParametersPresentAndValid(paramList))
	{	// huh, did the user tried to check for syntax before using this method?
		throw af::InvalidOperationException(_T("Invalid or insufficient step block parameters."));
	}

	// just for sake of clarity
  axis::String solverTypeParamName = afdf::AxisInputLanguage::StepSyntax.SolverTypeParameterName;
  axis::String startTimeParamName = afdf::AxisInputLanguage::StepSyntax.StepStartTimeParameterName;
  axis::String endTimeParamName = afdf::AxisInputLanguage::StepSyntax.StepEndTimeParameterName;

	// get parameters values specific to us
	axis::String solverTypeName = paramList.GetParameterValue(solverTypeParamName).ToString();
	real startTime = 
    (real)static_cast<aslse::NumberValue&>(paramList.GetParameterValue(startTimeParamName)).GetDouble();
	real endTime = 
    (real)static_cast<aslse::NumberValue&>(paramList.GetParameterValue(endTimeParamName)).GetDouble();

	// extract solver-specific parameters
	aslse::ParameterList& solverParams = paramList.Clone();
	solverParams.Consume(solverTypeParamName)
		.Consume(startTimeParamName)
		.Consume(endTimeParamName);

	// if a clockwork was explicitly declared, get it parameters too
	if (paramList.IsDeclared(afdf::AxisInputLanguage::StepSyntax.ClockworkTypeParameterName))
	{
		isClockworkDeclared = true;
		clockworkTypeName = paramList.GetParameterValue(
                            afdf::AxisInputLanguage::StepSyntax.ClockworkTypeParameterName).ToString();
		clockworkParams = &aslse::ParameterList::FromParameterArray(
								        static_cast<aslse::ArrayValue&>(paramList.GetParameterValue(
									      afdf::AxisInputLanguage::StepSyntax.ClockworkParamListParameterName)));

		solverParams.Consume(afdf::AxisInputLanguage::StepSyntax.ClockworkTypeParameterName)
					      .Consume(afdf::AxisInputLanguage::StepSyntax.ClockworkParamListParameterName);
	}

  // get step name, if present
  if (paramList.IsDeclared(afdf::AxisInputLanguage::StepSyntax.StepTitleParameterName))
  {
    stepName = paramList.GetParameterValue(afdf::AxisInputLanguage::StepSyntax.StepTitleParameterName).
                  ToString();
    solverParams.Consume(afdf::AxisInputLanguage::StepSyntax.StepTitleParameterName);
  }

	// build parser now!
	aapps::BlockParser *p = NULL;
	if (isClockworkDeclared)
	{
		p = new aapps::StepParser(*this, stepName,
						                  *solverLocator_, *clockworkLocator_,
						                  solverTypeName, startTime, endTime, solverParams, 
						                  clockworkTypeName, *clockworkParams,
                              *collectorLocator_, *formatLocator_);
	}
	else
	{
		p = new aapps::StepParser(*this, stepName, *solverLocator_, solverTypeName, startTime, endTime, 
                              solverParams, *collectorLocator_, *formatLocator_);
	}

	// free resources
	solverParams.Destroy();
	if (clockworkParams != NULL) clockworkParams->Destroy();

	return *p;
}

const char * aafp::StepParserProvider::GetFeaturePath( void ) const
{
	return "axis.base.input.providers.StepParserProvider";
}

const char * aafp::StepParserProvider::GetFeatureName( void ) const
{
	return "StepParserProvider";
}

void aafp::StepParserProvider::DoOnPostProcessRegistration( asmg::GlobalProviderCatalog& rootManager )
{
	// once we registered, get a reference to the solver and clockwork locators
	solverLocator_ = &static_cast<aal::SolverFactoryLocator&>(
                      rootManager.GetProvider(asmg::ServiceLocator::GetSolverLocatorPath()));
	clockworkLocator_ = &static_cast<aal::ClockworkFactoryLocator&>(
                      rootManager.GetProvider(asmg::ServiceLocator::GetClockworkFactoryLocatorPath()));
  formatLocator_ = &static_cast<aal::WorkbookFactoryLocator&>(
                      rootManager.GetProvider(asmg::ServiceLocator::GetWorkbookFactoryLocatorPath()));
  collectorLocator_ = &static_cast<aal::CollectorFactoryLocator&>(
                      rootManager.GetProvider(asmg::ServiceLocator::GetCollectorFactoryLocatorPath()));
}

bool aafp::StepParserProvider::IsRequiredParametersPresentAndValid( const aslse::ParameterList& paramList ) const
{
	// first, check if required parameters exist
	if (!(paramList.IsDeclared(afdf::AxisInputLanguage::StepSyntax.SolverTypeParameterName) &&
    paramList.IsDeclared(afdf::AxisInputLanguage::StepSyntax.StepStartTimeParameterName) &&
    paramList.IsDeclared(afdf::AxisInputLanguage::StepSyntax.StepEndTimeParameterName)))
	{
		return false;
	}

	// then, check if their data types are valid...
	aslse::ParameterValue& solverTypeParam = 
          paramList.GetParameterValue(afdf::AxisInputLanguage::StepSyntax.SolverTypeParameterName);
	aslse::ParameterValue& startTimeParam = 
          paramList.GetParameterValue(afdf::AxisInputLanguage::StepSyntax.StepStartTimeParameterName);
	aslse::ParameterValue& endTimeParam = 
          paramList.GetParameterValue(afdf::AxisInputLanguage::StepSyntax.StepEndTimeParameterName);
	
	// they must be atomic values...
	if (!(solverTypeParam.IsAtomic() && startTimeParam.IsAtomic() && endTimeParam.IsAtomic()))
	{
		return false;
	}

	// and of respective atomic data types
	bool ok =	(static_cast<aslse::AtomicValue&>(solverTypeParam).IsId() || 
					   static_cast<aslse::AtomicValue&>(solverTypeParam).IsString()) &&	// solver type is an id or a string
				     (static_cast<aslse::AtomicValue&>(startTimeParam).IsNumeric() && 
					   static_cast<aslse::AtomicValue&>(endTimeParam).IsNumeric());		// start/end time is numeric
	if (!ok) return false;

	// if the optional parameters were passed (explicit clockwork declaration), validate them too
  bool cwParamOk = true, nameParamOk = true;
	if (paramList.IsDeclared(afdf::AxisInputLanguage::StepSyntax.ClockworkTypeParameterName))
	{
		if (!paramList.IsDeclared(afdf::AxisInputLanguage::StepSyntax.ClockworkParamListParameterName)) return false;

		// check if it is essentially an array of assignments
		aslse::ParameterValue& arrayParams = 
        paramList.GetParameterValue(afdf::AxisInputLanguage::StepSyntax.ClockworkParamListParameterName);
		if (!arrayParams.IsArray()) return false;
		aslse::ParameterList * cwParamList = NULL;
		cwParamOk = false;
		try
		{
			cwParamList = &aslse::ParameterList::FromParameterArray(static_cast<aslse::ArrayValue&>(arrayParams));
			cwParamOk = true;
		}
		catch (...)
		{
			// ok, it is not valid			
		}
		if (cwParamList != NULL) cwParamList->Destroy();
	}
  if (paramList.IsDeclared(afdf::AxisInputLanguage::StepSyntax.StepTitleParameterName))
  {
    // check if it is essentially an array of assignments
    aslse::ParameterValue& titleValParam = 
        paramList.GetParameterValue(afdf::AxisInputLanguage::StepSyntax.StepTitleParameterName);
    if (!titleValParam.IsAtomic()) return false;
    aslse::AtomicValue& atomVal = static_cast<aslse::AtomicValue&>(titleValParam);
    nameParamOk = (atomVal.IsId() || atomVal.IsString());
  }
	
	return cwParamOk && nameParamOk;
}
