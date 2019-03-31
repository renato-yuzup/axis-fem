#include "TextReportElementCollectorFactory_Pimpl.hpp"
#include <assert.h>
#include "application/factories/collectors/CollectorFactory.hpp"
#include "application/output/collectors/Direction3DState.hpp"
#include "application/output/collectors/Direction6DState.hpp"
#include "application/output/collectors/TextReportElementStressCollector.hpp"
#include "application/output/collectors/TextReportElementStrainCollector.hpp"
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


aslp::ParseResult aafc::TextReportElementCollectorFactory::Pimpl::TryParseAny( 
  const asli::InputIterator& begin, 
  const asli::InputIterator& end )
{
  TextReportElementCollectorParser::ElementCollectorParseResult result = Parser.Parse(begin, end);
  return result.GetParseResult();
}

aafc::CollectorBuildResult aafc::TextReportElementCollectorFactory::Pimpl::ParseAndBuildAny( 
  const asli::InputIterator& begin, 
  const asli::InputIterator& end, 
  const ada::NumericalModel& model, 
  aapc::ParseContext& context )
{
  TextReportElementCollectorParser::ElementCollectorParseResult result = Parser.Parse(begin, end);
  TextReportElementCollectorParser::ElementCollectorParseResult::CollectorType collectorType =
    result.GetCollectorType();
  aslp::ParseResult parseResult = result.GetParseResult();

  if (!parseResult.IsMatch())
  { // it was passed an invalid statement
    throw axis::foundation::ArgumentException();
  }

  // check that node set exists
  if (!model.ExistsElementSet(result.GetTargetSetName()))
  { // no, doesn't exist
    MarkUndefinedElementSet(result.GetTargetSetName(), context);
    return CollectorBuildResult(NULL, aafc::kGenericCollectorType, parseResult, true);
  }

  int directionsCount = 3;
  if (collectorType == TextReportElementCollectorParser::ElementCollectorParseResult::kStress ||
    collectorType == TextReportElementCollectorParser::ElementCollectorParseResult::kStrain)
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

aaoc::GenericCollector& aafc::TextReportElementCollectorFactory::Pimpl::BuildCollector( 
  TextReportElementCollectorParser::ElementCollectorParseResult::CollectorType collectorType, 
  const axis::String& targetSetName, const bool * directionsToCollect ) const
{
  aaoc::GenericCollector *c = NULL;
  switch (collectorType)
  {
  case TextReportElementCollectorParser::ElementCollectorParseResult::kStress:
    c = new aaoc::TextReportElementStressCollector(targetSetName, directionsToCollect);
    break;
  case TextReportElementCollectorParser::ElementCollectorParseResult::kStrain:
    c = new aaoc::TextReportElementStrainCollector(targetSetName, directionsToCollect);
    break;
  default:
    assert(!_T("Undefined collector type for spatial type data!"));
    break;
  }
  return *c;
}

void aafc::TextReportElementCollectorFactory::Pimpl::MarkUndefinedElementSet( const axis::String& setName, 
                                                                       aapc::ParseContext& context ) const
{
  aapc::SymbolTable& st = context.Symbols();
  if (!st.IsSymbolDefined(setName, aapc::SymbolTable::kElementSet) && 
    context.GetRunMode() == aapc::ParseContext::kInspectionMode)
  { // register missing node set
    axis::String msg = AXIS_ERROR_MSG_ELEMENTSET_NOT_FOUND;
    msg.replace(_T("%1"), setName);
    context.RegisterEvent(asmm::ErrorMessage(AXIS_ERROR_ID_ELEMENTSET_NOT_FOUND, msg));
    return;
  }

  st.AddCurrentRoundUnresolvedSymbol(setName, aapc::SymbolTable::kElementSet);
}
