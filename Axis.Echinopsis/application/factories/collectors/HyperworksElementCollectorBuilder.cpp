#include "stdafx.h"
#include "HyperworksElementCollectorBuilder.hpp"
#include "application/output/collectors/ElementStressCollector.hpp"
#include "application/output/collectors/ElementStrainCollector.hpp"
#include "application/output/collectors/ElementArtificialEnergyCollector.hpp"
#include "application/output/collectors/ElementPlasticStrainCollector.hpp"
#include "application/output/collectors/ElementEffectivePlasticStrainCollector.hpp"
#include "application/output/collectors/ElementDeformationGradientCollector.hpp"

namespace aafc = axis::application::factories::collectors;
namespace aaoc = axis::application::output::collectors;

aafc::HyperworksElementCollectorBuilder::HyperworksElementCollectorBuilder(void)
{
  // nothing to do here
}

aafc::HyperworksElementCollectorBuilder::~HyperworksElementCollectorBuilder(void)
{
  // nothing to do here
}

aaoc::ElementSetCollector& aafc::HyperworksElementCollectorBuilder::
  BuildStressCollector( const axis::String& targetSetName, 
  aaoc::XXDirectionState xxState, aaoc::YYDirectionState yyState, 
  aaoc::ZZDirectionState zzState, aaoc::YZDirectionState yzState, 
  aaoc::XZDirectionState xzState, aaoc::XYDirectionState xyState )
{
  String fieldName = GetFieldName(_T("Stress"), xxState, yyState, zzState, 
    yzState, xzState, xyState);
  return aaoc::ElementStressCollector::Create(targetSetName, fieldName, 
    xxState, yyState, zzState, yzState, xzState, xyState);
}

aaoc::ElementSetCollector& aafc::HyperworksElementCollectorBuilder::
  BuildStrainCollector( const axis::String& targetSetName, 
  aaoc::XXDirectionState xxState, aaoc::YYDirectionState yyState, 
  aaoc::ZZDirectionState zzState, aaoc::YZDirectionState yzState, 
  aaoc::XZDirectionState xzState, aaoc::XYDirectionState xyState )
{
  String fieldName = GetFieldName(_T("Strain"), xxState, yyState, zzState, 
    yzState, xzState, xyState);
  return aaoc::ElementStrainCollector::Create(targetSetName, fieldName, 
    xxState, yyState, zzState, yzState, xzState, xyState);
}

aaoc::ElementSetCollector& aafc::HyperworksElementCollectorBuilder::
  BuildPlasticStrainIncrementCollector( const axis::String& targetSetName, 
  aaoc::XXDirectionState xxState, aaoc::YYDirectionState yyState, 
  aaoc::ZZDirectionState zzState, aaoc::YZDirectionState yzState, 
  aaoc::XZDirectionState xzState, aaoc::XYDirectionState xyState )
{
  String fieldName = GetFieldName(_T("Plastic Strain Increment"), xxState, 
    yyState, zzState, yzState, xzState, xyState);
  return aaoc::ElementPlasticStrainCollector::Create(targetSetName, 
    fieldName, xxState, yyState, zzState, yzState, xzState, xyState);
}

aaoc::ElementSetCollector& aafc::HyperworksElementCollectorBuilder::
  BuildArtificialEnergyCollector( const axis::String& targetSetName )
{
  return aaoc::ElementArtificialEnergyCollector::Create(targetSetName, 
    _T("Artificial Energy(s)"));
}

aaoc::ElementSetCollector& aafc::HyperworksElementCollectorBuilder::
  BuildEffectivePlasticStrainCollector( const axis::String& targetSetName )
{
  return aaoc::ElementEffectivePlasticStrainCollector::Create(targetSetName, 
    _T("Effective Plastic Strain(s)"));  
}

aaoc::ElementSetCollector& aafc::HyperworksElementCollectorBuilder::
  BuildDeformationGradientCollector( const axis::String& targetSetName )
{
  return aaoc::ElementDeformationGradientCollector::Create(targetSetName,
    _T("Deformation Gradient(t)"));
}

axis::String aafc::HyperworksElementCollectorBuilder::GetFieldName( 
  const axis::String& baseName, aaoc::XXDirectionState xxState, 
  aaoc::YYDirectionState yyState, aaoc::ZZDirectionState zzState, 
  aaoc::YZDirectionState yzState, aaoc::XZDirectionState xzState, 
  aaoc::XYDirectionState xyState ) const
{
  axis::String fieldName;
  if (xxState == aaoc::kXXEnabled && yyState == aaoc::kYYEnabled && 
    zzState == aaoc::kZZEnabled && xyState == aaoc::kXYEnabled && 
    yzState == aaoc::kYZEnabled && xzState == aaoc::kXZEnabled)
  {
    fieldName = baseName + _T("(t)");
  }
  else
  {
    if (xxState == aaoc::kXXEnabled) fieldName = baseName + _T(" XX(s)");
    if (yyState == aaoc::kYYEnabled) fieldName += (fieldName.empty()? 
      _T("") : _T(", ")) + baseName + _T(" YY(s)");
    if (zzState == aaoc::kZZEnabled) fieldName += (fieldName.empty()? 
      _T("") : _T(", ")) + baseName + _T(" ZZ(s)");    
    if (yzState == aaoc::kYZEnabled) fieldName += (fieldName.empty()? 
      _T("") : _T(", ")) + baseName + _T(" YZ(s)");
    if (xzState == aaoc::kXZEnabled) fieldName += (fieldName.empty()? 
      _T("") : _T(", ")) + baseName + _T(" XZ(s)");    
    if (xyState == aaoc::kXYEnabled) fieldName += (fieldName.empty()? 
      _T("") : _T(", ")) + baseName + _T(" XY(s)");
  }
  return fieldName;
}
