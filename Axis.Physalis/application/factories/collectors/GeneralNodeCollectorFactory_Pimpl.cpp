#include "GeneralNodeCollectorFactory_Pimpl.hpp"
#include <assert.h>
#include "application/output/collectors/Direction3DState.hpp"
#include "application/output/collectors/Direction6DState.hpp"
#include "application/output/collectors/NodeAccelerationCollector.hpp"
#include "application/output/collectors/NodeDisplacementCollector.hpp"
#include "application/output/collectors/NodeExternalLoadCollector.hpp"
#include "application/output/collectors/NodeStrainCollector.hpp"
#include "application/output/collectors/NodeStressCollector.hpp"
#include "application/output/collectors/NodeReactionForceCollector.hpp"
#include "application/output/collectors/NodeVelocityCollector.hpp"
#include "application/parsing/core/ParseContext.hpp"
#include "application/parsing/core/SymbolTable.hpp"
#include "application/parsing/error_messages.hpp"
#include "domain/analyses/NumericalModel.hpp"
#include "services/messaging/ErrorMessage.hpp"
#include "foundation/ArgumentException.hpp"
#include "NodeCollectorBuilder.hpp"

namespace aafc = axis::application::factories::collectors;
namespace aaoc = axis::application::output::collectors;
namespace aaocs = axis::application::output::collectors::summarizers;
namespace aapc = axis::application::parsing::core;
namespace aapc = axis::application::parsing::core;
namespace ada = axis::domain::analyses;
namespace asli = axis::services::language::iterators;
namespace aslp = axis::services::language::parsing;
namespace asmm = axis::services::messaging;


aslp::ParseResult aafc::GeneralNodeCollectorFactory::Pimpl::TryParseAny( 
                                                        const asli::InputIterator& begin, 
                                                        const asli::InputIterator& end )
{
  aafc::CollectorParseResult result = Parser.Parse(begin, end);
  if (result.GetParseResult().IsMatch())
  { // check if match goes for the required collector
    if (result.GetGroupingType() != aaocs::kNone)
    { // no, say that we failed
      result.GetParseResult().SetResult(aslp::ParseResult::FailedMatch);
    }
  }
  return result.GetParseResult();
}

aafc::CollectorBuildResult aafc::GeneralNodeCollectorFactory::Pimpl::ParseAndBuildAny( 
                                                        const asli::InputIterator& begin, 
                                                        const asli::InputIterator& end, 
                                                        const ada::NumericalModel& model, 
                                                        aapc::ParseContext& context,
                                                        NodeCollectorBuilder& builder)
{
  aafc::CollectorParseResult result = Parser.Parse(begin, end);
  aafc::CollectorType collectorType = result.GetCollectorType();
  aslp::ParseResult parseResult = result.GetParseResult();

  if (!parseResult.IsMatch() || result.GetGroupingType() != aaocs::kNone)
  { // it was passed an invalid statement
    throw axis::foundation::ArgumentException();
  }

  // check that node set exists
  if (!result.DoesActOnWholeModel())
  {
    if (!model.ExistsNodeSet(result.GetTargetSetName()))
    { // no, doesn't exist
      MarkUndefinedNodeSet(result.GetTargetSetName(), context);
      return CollectorBuildResult(NULL, aafc::kNodeCollectorType, parseResult, true);
    }
  }

  int directionsCount = 3;
  if (collectorType == aafc::kStress || collectorType == aafc::kStrain)
  {
    directionsCount = 6;
  }
  bool *directionsToCollect = new bool[directionsCount];
  for (int i = 0; i < directionsCount; ++i) 
  {
    directionsToCollect[i] = result.ShouldCollectDirection(i);
  }

  axis::String targetSetName = result.GetTargetSetName();
  aaoc::NodeSetCollector *c = &BuildCollector(collectorType, targetSetName, 
                                              directionsToCollect, builder);  

  return aafc::CollectorBuildResult(c, aafc::kNodeCollectorType, parseResult);
}

aaoc::NodeSetCollector& aafc::GeneralNodeCollectorFactory::Pimpl::BuildCollector( 
      aafc::CollectorType collectorType, const axis::String& targetSetName, 
      const bool * directionsToCollect, NodeCollectorBuilder& builder ) const
{
  aaoc::NodeSetCollector *c = NULL;
  if (collectorType == aafc::kStress || collectorType == aafc::kStrain)
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
    default:
      assert(!_T("Undefined collector type for tensor type data!"));
      break;
    }
  }
  else
  {
    aaoc::XDirectionState xState;
    aaoc::YDirectionState yState;
    aaoc::ZDirectionState zState;
    xState = directionsToCollect[0]? aaoc::kXEnabled : aaoc::kXDisabled;
    yState = directionsToCollect[1]? aaoc::kYEnabled : aaoc::kYDisabled;
    zState = directionsToCollect[2]? aaoc::kZEnabled : aaoc::kZDisabled;
    switch (collectorType)
    {
    case aafc::kDisplacement:
      c = &builder.BuildDisplacementCollector(targetSetName, xState, yState, zState);
      break;
    case aafc::kVelocity:
      c = &builder.BuildVelocityCollector(targetSetName, xState, yState, zState);
      break;
    case aafc::kAcceleration:
      c = &builder.BuildAccelerationCollector(targetSetName, xState, yState, zState);
      break;
    case aafc::kExternalLoad:
      c = &builder.BuildExternalLoadCollector(targetSetName, xState, yState, zState);
      break;
    case aafc::kReactionForce:
      c = &builder.BuildReactionForceCollector(targetSetName, xState, yState, zState);
      break;
    default:
      assert(!_T("Undefined collector type for spatial type data!"));
      break;
    }
  }
  return *c;
}

void aafc::GeneralNodeCollectorFactory::Pimpl::MarkUndefinedNodeSet( const axis::String& setName, 
                                                                      aapc::ParseContext& context ) const
{
  aapc::SymbolTable& st = context.Symbols();
  if (!st.IsSymbolDefined(setName, aapc::SymbolTable::kNodeSet) && 
      context.GetRunMode() == aapc::ParseContext::kInspectionMode)
  { // register missing node set
    axis::String msg = AXIS_ERROR_MSG_NODESET_NOT_FOUND;
    msg.replace(_T("%1"), setName);
    context.RegisterEvent(asmm::ErrorMessage(AXIS_ERROR_ID_NODESET_NOT_FOUND, msg));
    return;
  }

  st.AddCurrentRoundUnresolvedSymbol(setName, aapc::SymbolTable::kNodeSet);
}
