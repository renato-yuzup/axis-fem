#include "GeneralElementCollectorFactory_Pimpl.hpp"
#include <assert.h>
#include "ElementCollectorBuilder.hpp"
#include "application/output/collectors/Direction3DState.hpp"
#include "application/output/collectors/Direction6DState.hpp"
#include "application/output/collectors/ElementStrainCollector.hpp"
#include "application/output/collectors/ElementStressCollector.hpp"
#include "application/parsing/core/ParseContext.hpp"
#include "application/parsing/core/SymbolTable.hpp"
#include "application/parsing/error_messages.hpp"
#include "domain/analyses/NumericalModel.hpp"
#include "services/messaging/ErrorMessage.hpp"
#include "foundation/ArgumentException.hpp"

namespace aafc = axis::application::factories::collectors;
namespace aaoc = axis::application::output::collectors;
namespace aaocs = axis::application::output::collectors::summarizers;
namespace aapc = axis::application::parsing::core;
namespace aapc = axis::application::parsing::core;
namespace ada = axis::domain::analyses;
namespace asli = axis::services::language::iterators;
namespace aslp = axis::services::language::parsing;
namespace asmm = axis::services::messaging;

aslp::ParseResult aafc::GeneralElementCollectorFactory::Pimpl::TryParseAny( 
  const asli::InputIterator& begin, 
  const asli::InputIterator& end )
{
  aafc::CollectorParseResult result = parser_.Parse(begin, end);
  aslp::ParseResult parseResult = result.GetParseResult();
  if (parseResult.IsMatch())
  { // check if match goes for the required collector
    if (result.GetGroupingType() != aaocs::kNone)
    { // no, say that we failed
      parseResult.SetResult(aslp::ParseResult::FailedMatch);
    }
  }
  return parseResult;
}

aafc::CollectorBuildResult aafc::GeneralElementCollectorFactory::Pimpl::ParseAndBuildAny( 
  const asli::InputIterator& begin, 
  const asli::InputIterator& end, 
  const ada::NumericalModel& model, 
  aapc::ParseContext& context,
  ElementCollectorBuilder& builder)
{
  aafc::CollectorParseResult result = parser_.Parse(begin, end);
  aafc::CollectorType collectorType = result.GetCollectorType();
  aslp::ParseResult parseResult = result.GetParseResult();

  if (!parseResult.IsMatch() || result.GetGroupingType() != aaocs::kNone)
  { // it was passed an invalid statement
    throw axis::foundation::ArgumentException();
  }

  // check that element set exists
  if (!result.DoesActOnWholeModel())
  {
    if (!model.ExistsElementSet(result.GetTargetSetName()))
    { // no, doesn't exist
      MarkUndefinedElementSet(result.GetTargetSetName(), context);
      return CollectorBuildResult(NULL, aafc::kElementCollectorType, parseResult, true);
    }
  }

  int directionsCount = result.GetDirectionCount();
  bool *directionsToCollect = new bool[directionsCount];
  for (int i = 0; i < directionsCount; ++i) 
  {
    directionsToCollect[i] = result.ShouldCollectDirection(i);
  }

  axis::String targetSetName = result.GetTargetSetName();
  aaoc::ElementSetCollector *c = &BuildCollector(collectorType, targetSetName, 
    directionsToCollect, builder);  

  return aafc::CollectorBuildResult(c, aafc::kElementCollectorType, parseResult);
}

aaoc::ElementSetCollector& aafc::GeneralElementCollectorFactory::Pimpl::BuildCollector( 
    aafc::CollectorType collectorType, const axis::String& targetSetName, 
    const bool * directionsToCollect, ElementCollectorBuilder& builder ) const
{
  aaoc::ElementSetCollector *c = NULL;
  if (collectorType == aafc::kStress || collectorType == aafc::kStrain|| 
      collectorType == aafc::kPlasticStrain)
  {
    aaoc::XXDirectionState xxState;
    aaoc::YYDirectionState yyState;
    aaoc::ZZDirectionState zzState;
    aaoc::XYDirectionState xyState;
    aaoc::YZDirectionState yzState;
    aaoc::XZDirectionState xzState;
    xxState = directionsToCollect[0]? aaoc::kXXEnabled : aaoc::kXXDisabled;
    yyState = directionsToCollect[1]? aaoc::kYYEnabled : aaoc::kYYDisabled;
    zzState = directionsToCollect[2]? aaoc::kZZEnabled : aaoc::kZZDisabled;
    yzState = directionsToCollect[3]? aaoc::kYZEnabled : aaoc::kYZDisabled;
    xzState = directionsToCollect[4]? aaoc::kXZEnabled : aaoc::kXZDisabled;
    xyState = directionsToCollect[5]? aaoc::kXYEnabled : aaoc::kXYDisabled;
    switch (collectorType)
    {
    case aafc::kStress:
      c = &builder.BuildStressCollector(targetSetName, xxState, yyState, 
        zzState, yzState, xzState, xyState);
      break;
    case aafc::kStrain:
      c = &builder.BuildStrainCollector(targetSetName, xxState, yyState, 
        zzState, yzState, xzState, xyState);
      break;
    case aafc::kPlasticStrain:
      c = &builder.BuildPlasticStrainIncrementCollector(targetSetName, 
        xxState, yyState, zzState, yzState, xzState, xyState);
      break;
    default:
      assert(!_T("Undefined collector type for tensor type data!"));
      break;
    }
  }
  else if (collectorType == aafc::kArtificialEnergy)
  {
    c = &builder.BuildArtificialEnergyCollector(targetSetName);
  }
  else if (collectorType == aafc::kEffectivePlasticStrain)
  {
    c = &builder.BuildEffectivePlasticStrainCollector(targetSetName);
  }
  else if (collectorType == aafc::kDeformationGradient)
  {
    c = &builder.BuildDeformationGradientCollector(targetSetName);
  }
  else
  {
    assert(!_T("3D Element Collector type not implemented yet!!"));
  }
  return *c;
}

void aafc::GeneralElementCollectorFactory::Pimpl::MarkUndefinedElementSet( const axis::String& setName, 
                                                                     aapc::ParseContext& context ) const
{
  aapc::SymbolTable& st = context.Symbols();
  if (!st.IsSymbolDefined(setName, aapc::SymbolTable::kElementSet) && 
      context.GetRunMode() == aapc::ParseContext::kInspectionMode)
  { // register missing element set
    axis::String msg = AXIS_ERROR_MSG_ELEMENTSET_NOT_FOUND;
    msg.replace(_T("%1"), setName);
    context.RegisterEvent(asmm::ErrorMessage(AXIS_ERROR_ID_ELEMENTSET_NOT_FOUND, msg));
    return;
  }

  st.AddCurrentRoundUnresolvedSymbol(setName, aapc::SymbolTable::kElement);
}
