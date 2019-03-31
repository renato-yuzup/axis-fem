#pragma once
#include "application/factories/collectors/NodeCollectorBuilder.hpp"

namespace axis { namespace application { namespace factories { namespace collectors {

class HyperworksNodeCollectorBuilder : public NodeCollectorBuilder
{
public:
  HyperworksNodeCollectorBuilder(void);
  ~HyperworksNodeCollectorBuilder(void);

  virtual axis::application::output::collectors::NodeSetCollector& BuildDisplacementCollector( 
      const axis::String& targetSetName, axis::application::output::collectors::XDirectionState xState, 
      axis::application::output::collectors::YDirectionState yState, 
      axis::application::output::collectors::ZDirectionState zState );

  virtual axis::application::output::collectors::NodeSetCollector& BuildAccelerationCollector( 
      const axis::String& targetSetName, axis::application::output::collectors::XDirectionState xState, 
      axis::application::output::collectors::YDirectionState yState, 
      axis::application::output::collectors::ZDirectionState zState );

  virtual axis::application::output::collectors::NodeSetCollector& BuildVelocityCollector( 
      const axis::String& targetSetName, axis::application::output::collectors::XDirectionState xState, 
      axis::application::output::collectors::YDirectionState yState, 
      axis::application::output::collectors::ZDirectionState zState );

  virtual axis::application::output::collectors::NodeSetCollector& BuildExternalLoadCollector( 
      const axis::String& targetSetName, axis::application::output::collectors::XDirectionState xState, 
      axis::application::output::collectors::YDirectionState yState, 
      axis::application::output::collectors::ZDirectionState zState );

  virtual axis::application::output::collectors::NodeSetCollector& BuildReactionForceCollector( 
      const axis::String& targetSetName, axis::application::output::collectors::XDirectionState xState, 
      axis::application::output::collectors::YDirectionState yState, 
      axis::application::output::collectors::ZDirectionState zState );

  virtual axis::application::output::collectors::NodeSetCollector& BuildStressCollector( 
      const axis::String& targetSetName, axis::application::output::collectors::XXDirectionState xxState, 
      axis::application::output::collectors::YYDirectionState yyState, 
      axis::application::output::collectors::ZZDirectionState zzState, 
      axis::application::output::collectors::YZDirectionState yzState, 
      axis::application::output::collectors::XZDirectionState xzState, 
      axis::application::output::collectors::XYDirectionState xyState );

  virtual axis::application::output::collectors::NodeSetCollector& BuildStrainCollector( 
      const axis::String& targetSetName, axis::application::output::collectors::XXDirectionState xxState, 
      axis::application::output::collectors::YYDirectionState yyState, 
      axis::application::output::collectors::ZZDirectionState zzState, 
      axis::application::output::collectors::YZDirectionState yzState, 
      axis::application::output::collectors::XZDirectionState xzState, 
      axis::application::output::collectors::XYDirectionState xyState );
private:
  axis::String GetFieldName(const axis::String& baseName, 
                            axis::application::output::collectors::XDirectionState xState, 
                            axis::application::output::collectors::YDirectionState yState, 
                            axis::application::output::collectors::ZDirectionState zState) const;
  axis::String GetFieldName(const axis::String& baseName, 
                            axis::application::output::collectors::XXDirectionState xxState, 
                            axis::application::output::collectors::YYDirectionState yyState, 
                            axis::application::output::collectors::ZZDirectionState zzState, 
                            axis::application::output::collectors::YZDirectionState yzState, 
                            axis::application::output::collectors::XZDirectionState xzState, 
                            axis::application::output::collectors::XYDirectionState xyState) const;
};

} } } } // namespace axis::application::factories::collectors
