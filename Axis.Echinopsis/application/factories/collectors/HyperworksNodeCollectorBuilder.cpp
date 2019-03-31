#include "stdafx.h"
#include "HyperworksNodeCollectorBuilder.hpp"
#include "application/output/collectors/NodeAccelerationCollector.hpp"
#include "application/output/collectors/NodeDisplacementCollector.hpp"
#include "application/output/collectors/NodeExternalLoadCollector.hpp"
#include "application/output/collectors/NodeReactionForceCollector.hpp"
#include "application/output/collectors/NodeStrainCollector.hpp"
#include "application/output/collectors/NodeStressCollector.hpp"
#include "application/output/collectors/NodeVelocityCollector.hpp"

namespace aafc = axis::application::factories::collectors;
namespace aaoc = axis::application::output::collectors;

aafc::HyperworksNodeCollectorBuilder::HyperworksNodeCollectorBuilder( void )
{
  // nothing to do here
}

aafc::HyperworksNodeCollectorBuilder::~HyperworksNodeCollectorBuilder( void )
{
  // nothing to do here
}

aaoc::NodeSetCollector& aafc::HyperworksNodeCollectorBuilder::BuildDisplacementCollector( 
    const axis::String& targetSetName, aaoc::XDirectionState xState, aaoc::YDirectionState yState, 
    aaoc::ZDirectionState zState )
{
  axis::String fieldName = GetFieldName(_T("Displacement"), xState, yState, zState);
  return aaoc::NodeDisplacementCollector::Create(targetSetName, fieldName, xState, yState, zState);
}

aaoc::NodeSetCollector& aafc::HyperworksNodeCollectorBuilder::BuildAccelerationCollector( 
    const axis::String& targetSetName, aaoc::XDirectionState xState, aaoc::YDirectionState yState, 
    aaoc::ZDirectionState zState )
{
  axis::String fieldName = GetFieldName(_T("Acceleration"), xState, yState, zState);
  return aaoc::NodeAccelerationCollector::Create(targetSetName, fieldName, xState, yState, zState);
}

aaoc::NodeSetCollector& aafc::HyperworksNodeCollectorBuilder::BuildVelocityCollector( 
    const axis::String& targetSetName, aaoc::XDirectionState xState, aaoc::YDirectionState yState, 
    aaoc::ZDirectionState zState )
{
  axis::String fieldName = GetFieldName(_T("Velocity"), xState, yState, zState);
  return aaoc::NodeVelocityCollector::Create(targetSetName, fieldName, xState, yState, zState);
}

aaoc::NodeSetCollector& aafc::HyperworksNodeCollectorBuilder::BuildExternalLoadCollector( 
    const axis::String& targetSetName, aaoc::XDirectionState xState, aaoc::YDirectionState yState, 
    aaoc::ZDirectionState zState )
{
  axis::String fieldName = GetFieldName(_T("Force"), xState, yState, zState);
  return aaoc::NodeExternalLoadCollector::Create(targetSetName, fieldName, xState, yState, zState);
}

aaoc::NodeSetCollector& aafc::HyperworksNodeCollectorBuilder::BuildReactionForceCollector( 
    const axis::String& targetSetName, aaoc::XDirectionState xState, aaoc::YDirectionState yState, 
    aaoc::ZDirectionState zState )
{
  axis::String fieldName = GetFieldName(_T("Reaction"), xState, yState, zState);
  return aaoc::NodeReactionForceCollector::Create(targetSetName, fieldName, xState, yState, zState);
}

aaoc::NodeSetCollector& aafc::HyperworksNodeCollectorBuilder::BuildStressCollector( 
    const axis::String& targetSetName, aaoc::XXDirectionState xxState, aaoc::YYDirectionState yyState, 
    aaoc::ZZDirectionState zzState, aaoc::YZDirectionState yzState, 
    aaoc::XZDirectionState xzState, aaoc::XYDirectionState xyState )
{
  axis::String fieldName = GetFieldName(_T("Stress"), xxState, yyState, zzState, yzState, xzState, 
                                        xyState);
  return aaoc::NodeStressCollector::Create(targetSetName, fieldName, xxState, yyState, 
                                           zzState, yzState, xzState, xyState);
}

aaoc::NodeSetCollector& aafc::HyperworksNodeCollectorBuilder::BuildStrainCollector( 
    const axis::String& targetSetName, aaoc::XXDirectionState xxState, aaoc::YYDirectionState yyState, 
    aaoc::ZZDirectionState zzState, aaoc::YZDirectionState yzState, 
    aaoc::XZDirectionState xzState, aaoc::XYDirectionState xyState )
{
  axis::String fieldName = GetFieldName(_T("Strain"), xxState, yyState, zzState, yzState, xzState, 
                                        xyState);
  return aaoc::NodeStrainCollector::Create(targetSetName, fieldName, xxState, yyState, 
                                           zzState, yzState, xzState, xyState);
}


axis::String aafc::HyperworksNodeCollectorBuilder::GetFieldName( const axis::String& baseName, 
                                                                 aaoc::XDirectionState xState, 
                                                                 aaoc::YDirectionState yState, 
                                                                 aaoc::ZDirectionState zState ) const
{
  axis::String fieldName;
  if (xState == aaoc::kXEnabled && yState == aaoc::kYEnabled && zState == aaoc::kZEnabled)
  {
    fieldName = baseName + _T("(v)");
  }
  else
  {
    if (xState == aaoc::kXEnabled) fieldName = baseName + _T(" X(s)");
    if (yState == aaoc::kYEnabled) fieldName += (fieldName.empty()? _T("") : _T(", ")) + baseName + _T(" Y(s)");
    if (zState == aaoc::kZEnabled) fieldName += (fieldName.empty()? _T("") : _T(", ")) + baseName + _T(" Z(s)");    
  }
  return fieldName;
}

axis::String aafc::HyperworksNodeCollectorBuilder::GetFieldName( const axis::String& baseName, 
                                                                 aaoc::XXDirectionState xxState, 
                                                                 aaoc::YYDirectionState yyState, 
                                                                 aaoc::ZZDirectionState zzState, 
                                                                 aaoc::YZDirectionState yzState, 
                                                                 aaoc::XZDirectionState xzState, 
                                                                 aaoc::XYDirectionState xyState ) const
{
  axis::String fieldName;
  if (xxState == aaoc::kXXEnabled && yyState == aaoc::kYYEnabled && zzState == aaoc::kZZEnabled &&
      xyState == aaoc::kXYEnabled && yzState == aaoc::kYZEnabled && xzState == aaoc::kXZEnabled)
  {
    fieldName = baseName + _T("(t)");
  }
  else
  {
    if (xxState == aaoc::kXXEnabled) fieldName = baseName + _T(" XX(s)");
    if (yyState == aaoc::kYYEnabled) fieldName += (fieldName.empty()? _T("") : _T(", ")) + baseName + _T(" YY(s)");
    if (zzState == aaoc::kZZEnabled) fieldName += (fieldName.empty()? _T("") : _T(", ")) + baseName + _T(" ZZ(s)");    
    if (yzState == aaoc::kYZEnabled) fieldName += (fieldName.empty()? _T("") : _T(", ")) + baseName + _T(" YZ(s)");
    if (xzState == aaoc::kXZEnabled) fieldName += (fieldName.empty()? _T("") : _T(", ")) + baseName + _T(" XZ(s)");    
    if (xyState == aaoc::kXYEnabled) fieldName += (fieldName.empty()? _T("") : _T(", ")) + baseName + _T(" XY(s)");
  }
  return fieldName;
}
