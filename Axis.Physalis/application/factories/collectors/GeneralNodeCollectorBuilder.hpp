#pragma once
#include "NodeCollectorBuilder.hpp"

namespace axis { namespace application { namespace factories { namespace collectors {

/**
 * Represents an object that holds knowledge to build node collectors.
 */
class GeneralNodeCollectorBuilder : public NodeCollectorBuilder
{
public:
  virtual ~GeneralNodeCollectorBuilder(void);

  /**
   * Builds a displacement collector.
   *
   * @param targetSetName Name of the target set.
   * @param xState        Tells if X-direction value should be collected.
   * @param yState        Tells if Y-direction value should be collected.
   * @param zState        Tells if Z-direction value should be collected.
   *
   * @return The collector.
   */
  virtual axis::application::output::collectors::NodeSetCollector& BuildDisplacementCollector(
      const axis::String& targetSetName, 
      axis::application::output::collectors::XDirectionState xState, 
      axis::application::output::collectors::YDirectionState yState, 
      axis::application::output::collectors::ZDirectionState zState);

  /**
   * Builds an acceleration collector.
   *
   * @param targetSetName Name of the target set.
   * @param xState        Tells if X-direction value should be collected.
   * @param yState        Tells if Y-direction value should be collected.
   * @param zState        Tells if Z-direction value should be collected.
   *
   * @return The collector.
   */
  virtual axis::application::output::collectors::NodeSetCollector& BuildAccelerationCollector(
      const axis::String& targetSetName, 
      axis::application::output::collectors::XDirectionState xState, 
      axis::application::output::collectors::YDirectionState yState, 
      axis::application::output::collectors::ZDirectionState zState);

  /**
   * Builds a velocity collector.
   *
   * @param targetSetName Name of the target set.
   * @param xState        Tells if X-direction value should be collected.
   * @param yState        Tells if Y-direction value should be collected.
   * @param zState        Tells if Z-direction value should be collected.
   *
   * @return The collector.
   */
  virtual axis::application::output::collectors::NodeSetCollector& BuildVelocityCollector(
      const axis::String& targetSetName, 
      axis::application::output::collectors::XDirectionState xState, 
      axis::application::output::collectors::YDirectionState yState, 
      axis::application::output::collectors::ZDirectionState zState);

  /**
   * Builds an external load collector.
   *
   * @param targetSetName Name of the target set.
   * @param xState        Tells if X-direction value should be collected.
   * @param yState        Tells if Y-direction value should be collected.
   * @param zState        Tells if Z-direction value should be collected.
   *
   * @return The collector.
   */
  virtual axis::application::output::collectors::NodeSetCollector& BuildExternalLoadCollector(
      const axis::String& targetSetName, 
      axis::application::output::collectors::XDirectionState xState, 
      axis::application::output::collectors::YDirectionState yState, 
      axis::application::output::collectors::ZDirectionState zState);

  /**
   * Builds a reaction force collector.
   *
   * @param targetSetName Name of the target set.
   * @param xState        Tells if X-direction value should be collected.
   * @param yState        Tells if Y-direction value should be collected.
   * @param zState        Tells if Z-direction value should be collected.
   *
   * @return The collector.
   */
  virtual axis::application::output::collectors::NodeSetCollector& BuildReactionForceCollector(
      const axis::String& targetSetName, 
      axis::application::output::collectors::XDirectionState xState, 
      axis::application::output::collectors::YDirectionState yState, 
      axis::application::output::collectors::ZDirectionState zState);

  /**
   * Builds a stress collector.
   *
   * @param targetSetName Name of the target set.
   * @param xxState        Tells if XX-direction value should be collected.
   * @param yyState        Tells if YY-direction value should be collected.
   * @param zzState        Tells if ZZ-direction value should be collected.
   * @param yzState        Tells if YZ-direction value should be collected.
   * @param xzState        Tells if XZ-direction value should be collected.
   * @param xyState        Tells if XY-direction value should be collected.
   *
   * @return The collector.
   */
  virtual axis::application::output::collectors::NodeSetCollector& BuildStressCollector(
      const axis::String& targetSetName, 
      axis::application::output::collectors::XXDirectionState xxState, 
      axis::application::output::collectors::YYDirectionState yyState, 
      axis::application::output::collectors::ZZDirectionState zzState, 
      axis::application::output::collectors::YZDirectionState yzState, 
      axis::application::output::collectors::XZDirectionState xzState, 
      axis::application::output::collectors::XYDirectionState xyState);

  /**
   * Builds a strain collector.
   *
   * @param targetSetName Name of the target set.
   * @param xxState        Tells if XX-direction value should be collected.
   * @param yyState        Tells if YY-direction value should be collected.
   * @param zzState        Tells if ZZ-direction value should be collected.
   * @param yzState        Tells if YZ-direction value should be collected.
   * @param xzState        Tells if XZ-direction value should be collected.
   * @param xyState        Tells if XY-direction value should be collected.
   *
   * @return The collector.
   */
  virtual axis::application::output::collectors::NodeSetCollector& BuildStrainCollector(
      const axis::String& targetSetName, 
      axis::application::output::collectors::XXDirectionState xxState, 
      axis::application::output::collectors::YYDirectionState yyState, 
      axis::application::output::collectors::ZZDirectionState zzState, 
      axis::application::output::collectors::YZDirectionState yzState, 
      axis::application::output::collectors::XZDirectionState xzState, 
      axis::application::output::collectors::XYDirectionState xyState);
};

} } } } // namespace axis::application::factories::collectors
