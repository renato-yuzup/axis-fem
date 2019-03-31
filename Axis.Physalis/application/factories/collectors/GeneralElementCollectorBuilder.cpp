#include "GeneralElementCollectorBuilder.hpp"
#include "application/output/collectors/ElementArtificialEnergyCollector.hpp"
#include "application/output/collectors/ElementDeformationGradientCollector.hpp"
#include "application/output/collectors/ElementEffectivePlasticStrainCollector.hpp"
#include "application/output/collectors/ElementPlasticStrainCollector.hpp"
#include "application/output/collectors/ElementStrainCollector.hpp"
#include "application/output/collectors/ElementStressCollector.hpp"

namespace aafc = axis::application::factories::collectors;
namespace aaoc = axis::application::output::collectors;

aafc::GeneralElementCollectorBuilder::~GeneralElementCollectorBuilder( void )
{
  // nothing to do here
}

aaoc::ElementSetCollector& aafc::GeneralElementCollectorBuilder::
  BuildStressCollector( const axis::String& targetSetName, 
  aaoc::XXDirectionState xxState, aaoc::YYDirectionState yyState, 
  aaoc::ZZDirectionState zzState, aaoc::YZDirectionState yzState, 
  aaoc::XZDirectionState xzState, aaoc::XYDirectionState xyState )
{
  return aaoc::ElementStressCollector::Create(targetSetName, xxState, yyState, 
    zzState, yzState, xzState, xyState);
}

aaoc::ElementSetCollector& aafc::GeneralElementCollectorBuilder::
  BuildStrainCollector( const axis::String& targetSetName, 
  aaoc::XXDirectionState xxState, aaoc::YYDirectionState yyState, 
  aaoc::ZZDirectionState zzState, aaoc::YZDirectionState yzState, 
  aaoc::XZDirectionState xzState, aaoc::XYDirectionState xyState )
{
  return aaoc::ElementStrainCollector::Create(targetSetName, xxState, yyState, 
    zzState, yzState, xzState, xyState);
}

aaoc::ElementSetCollector& aafc::GeneralElementCollectorBuilder::
  BuildArtificialEnergyCollector( const axis::String& targetSetName )
{
  return aaoc::ElementArtificialEnergyCollector::Create(targetSetName);
}

aaoc::ElementSetCollector& aafc::GeneralElementCollectorBuilder::
  BuildPlasticStrainIncrementCollector( const axis::String& targetSetName, 
  aaoc::XXDirectionState xxState, aaoc::YYDirectionState yyState, 
  aaoc::ZZDirectionState zzState, aaoc::YZDirectionState yzState, 
  aaoc::XZDirectionState xzState, aaoc::XYDirectionState xyState )
{
  return aaoc::ElementPlasticStrainCollector::Create(targetSetName,
    xxState, yyState, zzState, yzState, xzState, xyState);
}

aaoc::ElementSetCollector& aafc::GeneralElementCollectorBuilder::
  BuildEffectivePlasticStrainCollector( const axis::String& targetSetName )
{
  return aaoc::ElementEffectivePlasticStrainCollector::Create(targetSetName);
}

aaoc::ElementSetCollector& aafc::GeneralElementCollectorBuilder::
  BuildDeformationGradientCollector( const axis::String& targetSetName )
{
  return aaoc::ElementDeformationGradientCollector::Create(targetSetName);
}
