#pragma once
#include "foundation/Axis.Physalis.hpp"
#include "AxisString.hpp"
#include "application/output/collectors/Direction3DState.hpp"
#include "application/output/collectors/Direction6DState.hpp"
#include "application/fwd/output_collectors.hpp"
#include "application/output/collectors/summarizers/SummaryType.hpp"
#include "SummaryElementCollectorBuilder.hpp"

namespace axis { namespace application { namespace factories { 
namespace collectors {

/**
 * Represents an object that holds knowledge to build element collectors.
 */
class AXISPHYSALIS_API GeneralSummaryElementCollectorBuilder : 
  public SummaryElementCollectorBuilder
{
public:
  virtual ~GeneralSummaryElementCollectorBuilder(void);

  /**
   * Builds a stress collector.
   *
   * @param targetSetName  Name of the target set.
   * @param summaryType    Tells how data will be summarized by the collector.
   * @param xxState        Tells if XX-direction value should be collected.
   * @param yyState        Tells if YY-direction value should be collected.
   * @param zzState        Tells if ZZ-direction value should be collected.
   * @param yzState        Tells if YZ-direction value should be collected.
   * @param xzState        Tells if XZ-direction value should be collected.
   * @param xyState        Tells if XY-direction value should be collected.
   *
   * @return The collector.
   */
  virtual axis::application::output::collectors::GenericCollector& 
    BuildStressCollector(const axis::String& targetSetName, 
    axis::application::output::collectors::summarizers::SummaryType summaryType,
    axis::application::output::collectors::XXDirectionState xxState, 
    axis::application::output::collectors::YYDirectionState yyState, 
    axis::application::output::collectors::ZZDirectionState zzState, 
    axis::application::output::collectors::YZDirectionState yzState, 
    axis::application::output::collectors::XZDirectionState xzState, 
    axis::application::output::collectors::XYDirectionState xyState);

  /**
   * Builds a strain collector.
   *
   * @param targetSetName  Name of the target set.
   * @param summaryType    Tells how data will be summarized by the collector.
   * @param xxState        Tells if XX-direction value should be collected.
   * @param yyState        Tells if YY-direction value should be collected.
   * @param zzState        Tells if ZZ-direction value should be collected.
   * @param yzState        Tells if YZ-direction value should be collected.
   * @param xzState        Tells if XZ-direction value should be collected.
   * @param xyState        Tells if XY-direction value should be collected.
   *
   * @return The collector.
   */
  virtual axis::application::output::collectors::GenericCollector& 
    BuildStrainCollector(const axis::String& targetSetName, 
    axis::application::output::collectors::summarizers::SummaryType summaryType,
    axis::application::output::collectors::XXDirectionState xxState, 
    axis::application::output::collectors::YYDirectionState yyState, 
    axis::application::output::collectors::ZZDirectionState zzState, 
    axis::application::output::collectors::YZDirectionState yzState, 
    axis::application::output::collectors::XZDirectionState xzState, 
    axis::application::output::collectors::XYDirectionState xyState);

  virtual axis::application::output::collectors::GenericCollector& 
    BuildPlasticStrainIncrementCollector( const axis::String& targetSetName, 
    axis::application::output::collectors::summarizers::SummaryType summaryType,
    axis::application::output::collectors::XXDirectionState xxState, 
    axis::application::output::collectors::YYDirectionState yyState, 
    axis::application::output::collectors::ZZDirectionState zzState, 
    axis::application::output::collectors::YZDirectionState yzState, 
    axis::application::output::collectors::XZDirectionState xzState, 
    axis::application::output::collectors::XYDirectionState xyState );

  /**
   * Builds a strain collector.
   *
   * @param targetSetName  Name of the target set.
   * @param summaryType    Tells how data will be summarized by the collector.
   *
   * @return The collector.
   */
  virtual axis::application::output::collectors::GenericCollector& 
    BuildArtificialEnergyCollector(const axis::String& targetSetName, 
    axis::application::output::collectors::summarizers::SummaryType summaryType);

  virtual axis::application::output::collectors::GenericCollector& 
    BuildEffectivePlasticStrainCollector( const axis::String& targetSetName,
    axis::application::output::collectors::summarizers::SummaryType summaryType);

  virtual axis::application::output::collectors::GenericCollector& 
    BuildDeformationGradientCollector( const axis::String& targetSetName,
    axis::application::output::collectors::summarizers::SummaryType summaryType);
};

} } } } // namespace axis::application::factories::collectors