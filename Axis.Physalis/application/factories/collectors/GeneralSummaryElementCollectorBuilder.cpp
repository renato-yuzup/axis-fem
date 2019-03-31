#include "GeneralSummaryElementCollectorBuilder.hpp"
#include "application/output/collectors/summarizers/SummaryElementArtificialEnergyCollector.hpp"
#include "application/output/collectors/summarizers/SummaryElementEffectivePlasticStrainCollector.hpp"
#include "application/output/collectors/summarizers/SummaryElementPlasticStrainCollector.hpp"
#include "application/output/collectors/summarizers/SummaryElementStrainCollector.hpp"
#include "application/output/collectors/summarizers/SummaryElementStressCollector.hpp"
#include "application/output/collectors/summarizers/SummaryElementDeformationGradientCollector.hpp"

namespace aafc = axis::application::factories::collectors;
namespace aaoc = axis::application::output::collectors;
namespace aaocs = aaoc::summarizers;

aafc::GeneralSummaryElementCollectorBuilder::~GeneralSummaryElementCollectorBuilder( void )
{
  // nothing to do here
}

aaoc::GenericCollector& aafc::GeneralSummaryElementCollectorBuilder::
  BuildStressCollector( const axis::String& targetSetName, 
  aaoc::summarizers::SummaryType summaryType, aaoc::XXDirectionState xxState, 
  aaoc::YYDirectionState yyState, aaoc::ZZDirectionState zzState, 
  aaoc::YZDirectionState yzState, aaoc::XZDirectionState xzState, 
  aaoc::XYDirectionState xyState )
{
  return aaocs::SummaryElementStressCollector::Create(targetSetName, 
    summaryType, xxState, yyState, zzState, yzState, xzState, xyState);
}

aaoc::GenericCollector& aafc::GeneralSummaryElementCollectorBuilder::
  BuildStrainCollector( const axis::String& targetSetName, 
  aaoc::summarizers::SummaryType summaryType, aaoc::XXDirectionState xxState, 
  aaoc::YYDirectionState yyState, aaoc::ZZDirectionState zzState, 
  aaoc::YZDirectionState yzState, aaoc::XZDirectionState xzState, 
  aaoc::XYDirectionState xyState )
{
  return aaocs::SummaryElementStrainCollector::Create(targetSetName, 
    summaryType, xxState, yyState, zzState, yzState, xzState, xyState);
}

aaoc::GenericCollector& aafc::GeneralSummaryElementCollectorBuilder::
  BuildPlasticStrainIncrementCollector( const axis::String& targetSetName,
  aaocs::SummaryType summaryType, aaoc::XXDirectionState xxState, 
  aaoc::YYDirectionState yyState, aaoc::ZZDirectionState zzState, 
  aaoc::YZDirectionState yzState, aaoc::XZDirectionState xzState, 
  aaoc::XYDirectionState xyState )
{
  return aaocs::SummaryElementPlasticStrainCollector::Create(
    targetSetName, summaryType, xxState, yyState, zzState, yzState, 
    xzState, xyState);
}

aaoc::GenericCollector& aafc::GeneralSummaryElementCollectorBuilder::
  BuildArtificialEnergyCollector( const axis::String& targetSetName, 
  aaocs::SummaryType summaryType )
{
  return aaocs::SummaryElementArtificialEnergyCollector::Create(targetSetName, 
    summaryType);
}

aaoc::GenericCollector& aafc::GeneralSummaryElementCollectorBuilder::
  BuildEffectivePlasticStrainCollector( const axis::String& targetSetName,
  aaocs::SummaryType summaryType)
{
  return aaocs::SummaryElementEffectivePlasticStrainCollector::Create(
    targetSetName, summaryType);
}

aaoc::GenericCollector& aafc::GeneralSummaryElementCollectorBuilder::
  BuildDeformationGradientCollector( const axis::String& targetSetName, 
  aaocs::SummaryType summaryType )
{
  return aaocs::SummaryElementDeformationGradientCollector::Create(
    targetSetName, summaryType);
}
