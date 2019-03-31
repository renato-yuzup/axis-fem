#include "StepParser.hpp"
#include "SnapshotParser.hpp"
#include "ResultCollectorParser.hpp"
#include "application/factories/parsers/StepParserProvider.hpp"
#include "application/jobs/AnalysisStep.hpp"
#include "application/jobs/StructuralAnalysis.hpp"
#include "application/locators/ClockworkFactoryLocator.hpp"
#include "application/locators/SolverFactoryLocator.hpp"
#include "application/output/ResultBucketConcrete.hpp"
#include "application/parsing/core/SymbolTable.hpp"
#include "application/parsing/core/ParseContext.hpp"
#include "application/parsing/parsers/EmptyBlockParser.hpp"
#include "application/parsing/error_messages.hpp"
#include "domain/algorithms/Solver.hpp"
#include "services/language/syntax/evaluation/ParameterList.hpp"
#include "services/messaging/ErrorMessage.hpp"
#include "foundation/NotSupportedException.hpp"
#include "foundation/definitions/AxisInputLanguage.hpp"

namespace aaj = axis::application::jobs;
namespace aal = axis::application::locators;
namespace aao = axis::application::output;
namespace aapps = axis::application::parsing::parsers;
namespace aafp = axis::application::factories::parsers;
namespace aapc = axis::application::parsing::core;
namespace adal = axis::domain::algorithms;
namespace asli = axis::services::language::iterators;
namespace aslse = axis::services::language::syntax::evaluation;
namespace aslp = axis::services::language::parsing;
namespace asmm = axis::services::messaging;
namespace afdf = axis::foundation::definitions;

// Enforce instantiation of this specialized template
template class aapps::StepParserTemplate<aal::SolverFactoryLocator, 
                                  aal::ClockworkFactoryLocator>;


template <class SolverFactoryLoc, class ClockworkFactoryLoc>
aapps::StepParserTemplate<SolverFactoryLoc, ClockworkFactoryLoc>::StepParserTemplate(
                                aafp::StepParserProvider& parentProvider, 
                                const axis::String& stepName,
                                SolverFactoryLoc& solverLocator, 
                                const axis::String& solverType, 
                                real startTime, real endTime, 
                                const aslse::ParameterList& solverParams,
                                aal::CollectorFactoryLocator& collectorLocator,
                                aal::WorkbookFactoryLocator& formatLocator) :
provider_(parentProvider), stepName_(stepName), solverLocator_(solverLocator), 
collectorLocator_(collectorLocator), formatLocator_(formatLocator)
{
  Init(solverType, startTime, endTime, solverParams);
}

template <class SolverFactoryLoc, class ClockworkFactoryLoc>
aapps::StepParserTemplate<SolverFactoryLoc, ClockworkFactoryLoc>::StepParserTemplate( 
                                aafp::StepParserProvider& parentProvider, 
                                const axis::String& stepName,
                                SolverFactoryLoc& solverLocator, 
                                ClockworkFactoryLoc& clockworkLocator, 
                                const axis::String& solverType, 
                                real startTime, real endTime, 
                                const aslse::ParameterList& solverParams, 
                                const axis::String& clockworkType, 
                                const aslse::ParameterList& clockworkParams,
                                aal::CollectorFactoryLocator& collectorLocator,
                                aal::WorkbookFactoryLocator& formatLocator) :
provider_(parentProvider), stepName_(stepName), solverLocator_(solverLocator), 
collectorLocator_(collectorLocator), formatLocator_(formatLocator)
{
  Init(solverType, startTime, endTime, solverParams);
  isClockworkDeclared_ = true;
  clockworkTypeName_ = clockworkType;
  clockworkParams_ = &clockworkParams.Clone();
  clockworkLocator_ = &clockworkLocator;
}

template <class SolverFactoryLoc, class ClockworkFactoryLoc>
void aapps::StepParserTemplate<SolverFactoryLoc, ClockworkFactoryLoc>::Init( 
                                const axis::String& solverType, real startTime, real endTime, 
                                const aslse::ParameterList& solverParams )
{
  solverTypeName_ = solverType;
  stepStartTime_ = startTime;
  stepEndTime_ = endTime;
  solverParams_ = &solverParams.Clone();
  isNewReadRound_ = false;
  dirtyStepBlock_ = false;
  isClockworkDeclared_ = false;
  clockworkParams_ = NULL;
  clockworkLocator_ = NULL;
  stepResultBucket_ = NULL;

  nullParser_ = new axis::application::parsing::parsers::EmptyBlockParser(provider_);
}

template <class SolverFactoryLoc, class ClockworkFactoryLoc>
aapps::StepParserTemplate<SolverFactoryLoc, ClockworkFactoryLoc>::~StepParserTemplate( void )
{
  delete nullParser_;
  solverParams_->Destroy();
  if (clockworkParams_ != NULL) clockworkParams_->Destroy();
  clockworkParams_ = NULL;
  solverParams_ = NULL;
}

template <class SolverFactoryLoc, class ClockworkFactoryLoc>
aapps::BlockParser& aapps::StepParserTemplate<SolverFactoryLoc, ClockworkFactoryLoc>::GetNestedContext( 
                                const axis::String& contextName, 
                                const aslse::ParameterList& paramList )
{
  // if couldn't build a step block, we cannot provide nested
  // contexts because it probably depends on the step begin
  // defined
  if (dirtyStepBlock_)
  {	// throwing this exception shows that we don't know how to 
    // proceed with these blocks
    throw axis::foundation::NotSupportedException();
  }

  // check if the requested context is the special context
  // 'SNAPSHOT'
  if (contextName == afdf::AxisInputLanguage::SnapshotsBlockName && paramList.IsEmpty())
  {	// yes, it is; let's override the default behavior and trick
    // the main parser giving him our own snapshot block parser
    SnapshotParser& p = *new SnapshotParser(isNewReadRound_);
    p.SetAnalysis(GetAnalysis());
    return p;
  }
  else if (contextName == afdf::AxisInputLanguage::ResultCollectionSyntax.BlockName)
  { // no, this is the special context 'OUTPUT'
    if (!ValidateCollectorBlockInformation(paramList))
    { // invalid, insufficient or unknown parameters
      throw axis::foundation::NotSupportedException();
    }
    BlockParser& p = CreateResultCollectorParser(paramList);
    p.SetAnalysis(GetAnalysis());
    return p;
  }

  // any other non-special block registered
  if (provider_.ContainsProvider(contextName, paramList))
  {
    aafp::BlockProvider& provider = provider_.GetProvider(contextName, paramList);
    aapps::BlockParser& nestedContext = provider.BuildParser(contextName, paramList);
    nestedContext.SetAnalysis(GetAnalysis());
    return nestedContext;
  }

  // no provider found
  throw axis::foundation::NotSupportedException();
}

template <class SolverFactoryLoc, class ClockworkFactoryLoc>
aslp::ParseResult aapps::StepParserTemplate<SolverFactoryLoc, ClockworkFactoryLoc>::Parse( 
                          const asli::InputIterator& begin, 
                          const asli::InputIterator& end )
{
  return nullParser_->Parse(begin, end);
}

template <class SolverFactoryLoc, class ClockworkFactoryLoc>
void aapps::StepParserTemplate<SolverFactoryLoc, ClockworkFactoryLoc>::DoStartContext( void )
{
  aapc::SymbolTable& st = GetParseContext().Symbols();
  nullParser_->StartContext(GetParseContext());

  // first, check if we are able to build the specified solver
  bool canBuildSolver;
  if (isClockworkDeclared_)
  {
    canBuildSolver = solverLocator_.CanBuild(solverTypeName_, *solverParams_, 
                                             stepStartTime_, stepEndTime_, 
                                             clockworkTypeName_, *clockworkParams_);
  }
  else
  {
    canBuildSolver = solverLocator_.CanBuild(solverTypeName_, *solverParams_, 
                                             stepStartTime_, stepEndTime_);
  }
  if (!canBuildSolver)
  {	// huh?
    GetParseContext().RegisterEvent(asmm::ErrorMessage(AXIS_ERROR_ID_UNKNOWN_SOLVER_TYPE, 
                                                       AXIS_ERROR_MSG_UNKNOWN_SOLVER_TYPE));
    dirtyStepBlock_ = true;
    return;
  }

  // then, check if this is a new read round
  axis::String symbolName = st.GenerateDecoratedName(aapc::SymbolTable::kAnalysisStep);

  if (st.IsSymbolDefined(symbolName, aapc::SymbolTable::kAnalysisStep))
  {	// yes, it is a new read round; we don't need to create a new solver and step objects
    isNewReadRound_ = true;
  }
  st.DefineOrRefreshSymbol(symbolName, aapc::SymbolTable::kAnalysisStep);

  // set analysis step to work on 
  aaj::StructuralAnalysis& analysis = GetAnalysis();
  if (!isNewReadRound_)
  {	// we need to create the new step

    // build clockwork if we have to
    adal::Solver *solver = NULL;
    if (isClockworkDeclared_)
    {
      adal::Clockwork *clockwork = NULL;
      if (clockworkLocator_->CanBuild(clockworkTypeName_, *clockworkParams_, 
                                      stepStartTime_, stepEndTime_))
      {
        clockwork = &clockworkLocator_->BuildClockwork(clockworkTypeName_, 
                                                       *clockworkParams_, 
                                                       stepStartTime_, 
                                                       stepEndTime_);
      }
      else
      {
        // there is something wrong with supplied parameters
        GetParseContext().RegisterEvent(
                asmm::ErrorMessage(AXIS_ERROR_ID_UNKNOWN_TIME_CONTROL_ALGORITHM, 
                                   AXIS_ERROR_MSG_UNKNOWN_TIME_CONTROL_ALGORITHM));
        dirtyStepBlock_ = true;
        return;
      }
      solver = &solverLocator_.BuildSolver(solverTypeName_, *solverParams_, 
                                           stepStartTime_, stepEndTime_, *clockwork);
    }
    else
    {
      solver = &solverLocator_.BuildSolver(solverTypeName_, *solverParams_, 
                                           stepStartTime_, stepEndTime_);
    }

    stepResultBucket_ = new aao::ResultBucketConcrete();
    aaj::AnalysisStep& step = aaj::AnalysisStep::Create(stepStartTime_, stepEndTime_, 
                                                        *solver, *stepResultBucket_);
    step.SetName(stepName_);
    analysis.AddStep(step);
  }
  else
  { // since we are on a new parse round, retrieve result bucket that we created earlier
    int stepIndex = GetParseContext().GetStepOnFocusIndex() + 1;
    aaj::AnalysisStep *step = &GetAnalysis().GetStep(stepIndex);
    stepResultBucket_ = static_cast<aao::ResultBucketConcrete *>(&step->GetResults());
    GetParseContext().SetStepOnFocus(step);
    GetParseContext().SetStepOnFocusIndex(stepIndex);
  }

  if (analysis.GetStepCount() == 1)
  {	// we are parsing the first step
    GetParseContext().SetStepOnFocus(&analysis.GetStep(0));
    GetParseContext().SetStepOnFocusIndex(0);
  }
  else
  {
    int nextStepIndex = GetParseContext().GetStepOnFocusIndex() + 1;
    GetParseContext().SetStepOnFocus(&analysis.GetStep(nextStepIndex));
    GetParseContext().SetStepOnFocusIndex(nextStepIndex);
  }
}

template <class SolverFactoryLoc, class ClockworkFactoryLoc>
bool aapps::StepParserTemplate<SolverFactoryLoc, ClockworkFactoryLoc>
                  ::ValidateCollectorBlockInformation( const aslse::ParameterList& contextParams )
{
  int paramsFound = 2;
  if (!contextParams.IsDeclared(
    afdf::AxisInputLanguage::ResultCollectionSyntax.FileNameParameterName)) return false;
  if (!contextParams.IsDeclared(
    afdf::AxisInputLanguage::ResultCollectionSyntax.FileFormatParameterName)) return false;

  // check optional parameters
  if (contextParams.IsDeclared(afdf::AxisInputLanguage::ResultCollectionSyntax.AppendParameterName))
  {
    String val = contextParams.GetParameterValue(
      afdf::AxisInputLanguage::ResultCollectionSyntax.AppendParameterName).ToString();
    val.to_lower_case().trim();
    if (val != _T("yes") && val != _T("no") && val != _T("true") && val != _T("false"))
    {
      return false;
    }
    paramsFound++;
  }
  if (contextParams.IsDeclared(
      afdf::AxisInputLanguage::ResultCollectionSyntax.FormatArgumentsParameterName))
  {
    aslse::ParameterValue& val = contextParams.GetParameterValue(
      afdf::AxisInputLanguage::ResultCollectionSyntax.FormatArgumentsParameterName);
    if (!val.IsArray()) return false;

    // check if it is an array of parameters
    try
    {
      aslse::ParameterList& test = 
        aslse::ParameterList::FromParameterArray(static_cast<aslse::ArrayValue&>(val));
      test.Destroy();
    }
    catch (...)
    {	// uh oh, it is not valid
      return false;
    }
    paramsFound++;
  }

  return (contextParams.Count() == paramsFound);
}

template <class SolverFactoryLoc, class ClockworkFactoryLoc>
aapps::BlockParser& aapps::StepParserTemplate<SolverFactoryLoc, ClockworkFactoryLoc>
  ::CreateResultCollectorParser( const aslse::ParameterList& contextParams ) const
{
  String fileName = contextParams.GetParameterValue(
                    afdf::AxisInputLanguage::ResultCollectionSyntax.FileNameParameterName).ToString();
  String formatName = contextParams.GetParameterValue(
                    afdf::AxisInputLanguage::ResultCollectionSyntax.FileFormatParameterName).ToString();
  const aslse::ParameterList *formatArgs = NULL;

  bool append = false;
  if (contextParams.IsDeclared(afdf::AxisInputLanguage::ResultCollectionSyntax.AppendParameterName))
  {
    String val = contextParams.GetParameterValue(
                        afdf::AxisInputLanguage::ResultCollectionSyntax.AppendParameterName).ToString();
    val.to_lower_case().trim();
    append = (val != _T("yes") && val != _T("true"));
  }
  if (contextParams.IsDeclared(
    afdf::AxisInputLanguage::ResultCollectionSyntax.FormatArgumentsParameterName))
  {
    aslse::ParameterValue& val = contextParams.GetParameterValue(
                          afdf::AxisInputLanguage::ResultCollectionSyntax.FormatArgumentsParameterName);
    formatArgs = &aslse::ParameterList::FromParameterArray(static_cast<aslse::ArrayValue&>(val));
  }
  else
  {
    formatArgs = &aslse::ParameterList::Empty.Clone();
  }

  aapps::BlockParser *parser = new aapps::ResultCollectorParser(formatLocator_, collectorLocator_, 
                                                                *stepResultBucket_, fileName, 
                                                                formatName, *formatArgs, append);
  formatArgs->Destroy();
  return *parser;
}
