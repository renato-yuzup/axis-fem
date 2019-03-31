#pragma once
#include "ElementCollectorBuilder.hpp"

namespace axis { namespace application { namespace factories { namespace collectors {

/**
 * Represents an object that holds knowledge to build element collectors.
 */
class GeneralElementCollectorBuilder : public ElementCollectorBuilder
{
public:
  virtual ~GeneralElementCollectorBuilder(void);

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
  virtual axis::application::output::collectors::ElementSetCollector& 
    BuildStressCollector(const axis::String& targetSetName, 
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
  virtual axis::application::output::collectors::ElementSetCollector& 
    BuildStrainCollector(const axis::String& targetSetName, 
    axis::application::output::collectors::XXDirectionState xxState, 
    axis::application::output::collectors::YYDirectionState yyState, 
    axis::application::output::collectors::ZZDirectionState zzState, 
    axis::application::output::collectors::YZDirectionState yzState, 
    axis::application::output::collectors::XZDirectionState xzState, 
    axis::application::output::collectors::XYDirectionState xyState);

  virtual axis::application::output::collectors::ElementSetCollector& 
    BuildPlasticStrainIncrementCollector( const axis::String& targetSetName, 
    axis::application::output::collectors::XXDirectionState xxState, 
    axis::application::output::collectors::YYDirectionState yyState, 
    axis::application::output::collectors::ZZDirectionState zzState, 
    axis::application::output::collectors::YZDirectionState yzState, 
    axis::application::output::collectors::XZDirectionState xzState, 
    axis::application::output::collectors::XYDirectionState xyState );

  /**
   * Builds an artificial energy collector.
   *
   * @param targetSetName Name of the target set.
   *
   * @return The collector.
   */
  virtual axis::application::output::collectors::ElementSetCollector& 
    BuildArtificialEnergyCollector(const axis::String& targetSetName);

  virtual axis::application::output::collectors::ElementSetCollector& 
    BuildEffectivePlasticStrainCollector( const axis::String& targetSetName );

  virtual axis::application::output::collectors::ElementSetCollector& 
    BuildDeformationGradientCollector( const axis::String& targetSetName );
};

} } } } // namespace axis::application::factories::collectors
