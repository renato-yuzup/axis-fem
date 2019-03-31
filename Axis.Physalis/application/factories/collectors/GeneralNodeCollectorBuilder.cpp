#include "GeneralNodeCollectorBuilder.hpp"
#include "application/output/collectors/NodeDisplacementCollector.hpp"
#include "application/output/collectors/NodeAccelerationCollector.hpp"
#include "application/output/collectors/NodeVelocityCollector.hpp"
#include "application/output/collectors/NodeExternalLoadCollector.hpp"
#include "application/output/collectors/NodeReactionForceCollector.hpp"
#include "application/output/collectors/NodeStressCollector.hpp"
#include "application/output/collectors/NodeStrainCollector.hpp"

namespace aafc = axis::application::factories::collectors;
namespace aaoc = axis::application::output::collectors;

aafc::GeneralNodeCollectorBuilder::~GeneralNodeCollectorBuilder( void )
{
  // nothing to do here
}

aaoc::NodeSetCollector& aafc::GeneralNodeCollectorBuilder::BuildDisplacementCollector( 
  const axis::String& targetSetName, aaoc::XDirectionState xState, 
  aaoc::YDirectionState yState, aaoc::ZDirectionState zState )
{
  return aaoc::NodeDisplacementCollector::Create(targetSetName, xState, yState, zState);
}

aaoc::NodeSetCollector& aafc::GeneralNodeCollectorBuilder::BuildAccelerationCollector( 
  const axis::String& targetSetName, aaoc::XDirectionState xState, 
  aaoc::YDirectionState yState, aaoc::ZDirectionState zState )
{
  return aaoc::NodeAccelerationCollector::Create(targetSetName, xState, yState, zState);
}

aaoc::NodeSetCollector& aafc::GeneralNodeCollectorBuilder::BuildVelocityCollector( 
  const axis::String& targetSetName, aaoc::XDirectionState xState, 
  aaoc::YDirectionState yState, aaoc::ZDirectionState zState )
{
  return aaoc::NodeVelocityCollector::Create(targetSetName, xState, yState, zState);
}

aaoc::NodeSetCollector& aafc::GeneralNodeCollectorBuilder::BuildExternalLoadCollector( 
  const axis::String& targetSetName, aaoc::XDirectionState xState, 
  aaoc::YDirectionState yState, aaoc::ZDirectionState zState )
{
  return aaoc::NodeExternalLoadCollector::Create(targetSetName, xState, yState, zState);
}

aaoc::NodeSetCollector& aafc::GeneralNodeCollectorBuilder::BuildReactionForceCollector( 
  const axis::String& targetSetName, aaoc::XDirectionState xState, 
  aaoc::YDirectionState yState, aaoc::ZDirectionState zState )
{
  return aaoc::NodeReactionForceCollector::Create(targetSetName, xState, yState, zState);
}

aaoc::NodeSetCollector& aafc::GeneralNodeCollectorBuilder::BuildStressCollector( 
  const axis::String& targetSetName, aaoc::XXDirectionState xxState, 
  aaoc::YYDirectionState yyState, aaoc::ZZDirectionState zzState, 
  aaoc::YZDirectionState yzState, aaoc::XZDirectionState xzState, 
  aaoc::XYDirectionState xyState )
{
  return aaoc::NodeStressCollector::Create(targetSetName, xxState, yyState, 
    zzState, yzState, xzState, xyState);
}

aaoc::NodeSetCollector& aafc::GeneralNodeCollectorBuilder::BuildStrainCollector( 
  const axis::String& targetSetName, aaoc::XXDirectionState xxState, 
  aaoc::YYDirectionState yyState, aaoc::ZZDirectionState zzState, 
  aaoc::YZDirectionState yzState, aaoc::XZDirectionState xzState, 
  aaoc::XYDirectionState xyState )
{
  return aaoc::NodeStrainCollector::Create(targetSetName, xxState, yyState, 
    zzState, yzState, xzState, xyState);
}
