#include "TextReportNodeCollectorFactory_Pimpl.hpp"
#include <assert.h>
#include "application/factories/collectors/CollectorFactory.hpp"
#include "application/output/collectors/Direction3DState.hpp"
#include "application/output/collectors/Direction6DState.hpp"
#include "application/output/collectors/TextReportNodeStressCollector.hpp"
#include "application/output/collectors/TextReportNodeStrainCollector.hpp"
#include "application/output/collectors/TextReportNodeReactionForceCollector.hpp"
#include "application/output/collectors/TextReportNodeAccelerationCollector.hpp"
#include "application/output/collectors/TextReportNodeVelocityCollector.hpp"
#include "application/output/collectors/TextReportNodeDisplacementCollector.hpp"
#include "application/output/collectors/TextReportNodeLoadCollector.hpp"
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


aslp::ParseResult aafc::TextReportNodeCollectorFactory::Pimpl::TryParseAny( 
  const asli::InputIterator& begin, 
  const asli::InputIterator& end )
{
  TextReportNodeCollectorParser::NodeCollectorParseResult result = Parser.Parse(begin, end);
  return result.GetParseResult();
}

aafc::CollectorBuildResult aafc::TextReportNodeCollectorFactory::Pimpl::ParseAndBuildAny( 
  const asli::InputIterator& begin, 
  const asli::InputIterator& end, 
  const ada::NumericalModel& model, 
  aapc::ParseContext& context )
{
  TextReportNodeCollectorParser::NodeCollectorParseResult result = Parser.Parse(begin, end);
  TextReportNodeCollectorParser::NodeCollectorParseResult::CollectorType collectorType =
    result.GetCollectorType();
  aslp::ParseResult parseResult = result.GetParseResult();

  if (!parseResult.IsMatch())
  { // it was passed an invalid statement
    throw axis::foundation::ArgumentException();
  }

  // check that node set exists
  if (!model.ExistsNodeSet(result.GetTargetSetName()))
  { // no, doesn't exist
    MarkUndefinedNodeSet(result.GetTargetSetName(), context);
    return CollectorBuildResult(NULL, aafc::kGenericCollectorType, parseResult, true);
  }

  int directionsCount = 3;
  if (collectorType == TextReportNodeCollectorParser::NodeCollectorParseResult::kStress ||
    collectorType == TextReportNodeCollectorParser::NodeCollectorParseResult::kStrain)
  {
    directionsCount = 6;
  }
  bool *directionsToCollect = new bool[directionsCount];
  for (int i = 0; i < directionsCount; ++i) 
  {
    directionsToCollect[i] = result.ShouldCollectDirection(i);
  }

  axis::String targetSetName = result.GetTargetSetName();
  aaoc::GenericCollector *c = &BuildCollector(collectorType, targetSetName, 
    directionsToCollect);  

  return aafc::CollectorBuildResult(c, aafc::kGenericCollectorType, parseResult);
}

aaoc::GenericCollector& aafc::TextReportNodeCollectorFactory::Pimpl::BuildCollector( 
  TextReportNodeCollectorParser::NodeCollectorParseResult::CollectorType collectorType, 
  const axis::String& targetSetName, const bool * directionsToCollect ) const
{
  aaoc::GenericCollector *c = NULL;
  switch (collectorType)
  {
  case TextReportNodeCollectorParser::NodeCollectorParseResult::kStress:
    c = new aaoc::TextReportNodeStressCollector(targetSetName, directionsToCollect);
    break;
  case TextReportNodeCollectorParser::NodeCollectorParseResult::kStrain:
    c = new aaoc::TextReportNodeStrainCollector(targetSetName, directionsToCollect);
    break;
  case TextReportNodeCollectorParser::NodeCollectorParseResult::kDisplacement:
    c = new aaoc::TextReportNodeDisplacementCollector(targetSetName, directionsToCollect);
    break;
  case TextReportNodeCollectorParser::NodeCollectorParseResult::kVelocity:
    c = new aaoc::TextReportNodeVelocityCollector(targetSetName, directionsToCollect);
    break;
  case TextReportNodeCollectorParser::NodeCollectorParseResult::kAcceleration:
    c = new aaoc::TextReportNodeAccelerationCollector(targetSetName, directionsToCollect);
    break;
  case TextReportNodeCollectorParser::NodeCollectorParseResult::kReactionForce:
    c = new aaoc::TextReportNodeReactionForceCollector(targetSetName, directionsToCollect);
    break;
  case TextReportNodeCollectorParser::NodeCollectorParseResult::kExternalLoad:
    c = new aaoc::TextReportNodeLoadCollector(targetSetName, directionsToCollect);
    break;
  default:
    assert(!_T("Undefined collector type for spatial type data!"));
    break;
  }
  return *c;
}

void aafc::TextReportNodeCollectorFactory::Pimpl::MarkUndefinedNodeSet( const axis::String& setName, 
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
