#include "GeneralSummaryNodeCollectorBuilder.hpp"
#include "application/output/collectors/summarizers/SummaryNodeStressCollector.hpp"
#include "application/output/collectors/summarizers/SummaryNodeStrainCollector.hpp"
#include "application/output/collectors/summarizers/SummaryNodeDisplacementCollector.hpp"
#include "application/output/collectors/summarizers/SummaryNodeExternalLoadCollector.hpp"
#include "application/output/collectors/summarizers/SummaryNodeVelocityCollector.hpp"
#include "application/output/collectors/summarizers/SummaryNodeAccelerationCollector.hpp"
#include "application/output/collectors/summarizers/SummaryNodeReactionForceCollector.hpp"

namespace aafc = axis::application::factories::collectors;
namespace aaoc = axis::application::output::collectors;
namespace aaocs = axis::application::output::collectors::summarizers;

aafc::GeneralSummaryNodeCollectorBuilder::~GeneralSummaryNodeCollectorBuilder( void )
{
  // nothing to do here
}

aaoc::GenericCollector& aafc::GeneralSummaryNodeCollectorBuilder::BuildDisplacementCollector( 
  const axis::String& targetSetName, aaoc::summarizers::SummaryType summaryType, 
  aaoc::XDirectionState xState, aaoc::YDirectionState yState, aaoc::ZDirectionState zState )
{
  return aaocs::SummaryNodeDisplacementCollector::Create(targetSetName, summaryType, xState, 
                                                         yState, zState);
}

aaoc::GenericCollector& aafc::GeneralSummaryNodeCollectorBuilder::BuildAccelerationCollector( 
  const axis::String& targetSetName, aaoc::summarizers::SummaryType summaryType, 
  aaoc::XDirectionState xState, aaoc::YDirectionState yState, aaoc::ZDirectionState zState )
{
  return aaocs::SummaryNodeAccelerationCollector::Create(targetSetName, summaryType, xState, 
                                                         yState, zState);
}

aaoc::GenericCollector& aafc::GeneralSummaryNodeCollectorBuilder::BuildVelocityCollector( 
  const axis::String& targetSetName, aaoc::summarizers::SummaryType summaryType, 
  aaoc::XDirectionState xState, aaoc::YDirectionState yState, aaoc::ZDirectionState zState )
{
  return aaocs::SummaryNodeVelocityCollector::Create(targetSetName, summaryType, xState, yState, zState);
}

aaoc::GenericCollector& aafc::GeneralSummaryNodeCollectorBuilder::BuildExternalLoadCollector( 
  const axis::String& targetSetName, aaoc::summarizers::SummaryType summaryType, 
  aaoc::XDirectionState xState, aaoc::YDirectionState yState, aaoc::ZDirectionState zState )
{
  return aaocs::SummaryNodeExternalLoadCollector::Create(targetSetName, summaryType, xState, 
                                                         yState, zState);
}

aaoc::GenericCollector& aafc::GeneralSummaryNodeCollectorBuilder::BuildReactionForceCollector( 
  const axis::String& targetSetName, aaoc::summarizers::SummaryType summaryType, 
  aaoc::XDirectionState xState, aaoc::YDirectionState yState, aaoc::ZDirectionState zState )
{
  return aaocs::SummaryNodeReactionForceCollector::Create(targetSetName, summaryType, xState, 
                                                          yState, zState);
}

aaoc::GenericCollector& aafc::GeneralSummaryNodeCollectorBuilder::BuildStressCollector( 
  const axis::String& targetSetName, aaoc::summarizers::SummaryType summaryType, 
  aaoc::XXDirectionState xxState, aaoc::YYDirectionState yyState, 
  aaoc::ZZDirectionState zzState, aaoc::YZDirectionState yzState, 
  aaoc::XZDirectionState xzState, aaoc::XYDirectionState xyState )
{
  return aaocs::SummaryNodeStressCollector::Create(targetSetName, summaryType, xxState, yyState, 
                                                   zzState, yzState, xzState, xyState);
}

aaoc::GenericCollector& aafc::GeneralSummaryNodeCollectorBuilder::BuildStrainCollector( 
  const axis::String& targetSetName, aaoc::summarizers::SummaryType summaryType, 
  aaoc::XXDirectionState xxState, aaoc::YYDirectionState yyState, 
  aaoc::ZZDirectionState zzState, aaoc::YZDirectionState yzState, 
  aaoc::XZDirectionState xzState, aaoc::XYDirectionState xyState )
{
  return aaocs::SummaryNodeStrainCollector::Create(targetSetName, summaryType, xxState, yyState, 
                                                   zzState, yzState, xzState, xyState);
}
