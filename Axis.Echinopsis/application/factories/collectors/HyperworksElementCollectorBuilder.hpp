#pragma once
#include "application/factories/collectors/ElementCollectorBuilder.hpp"

namespace axis { namespace application { namespace factories { namespace collectors {

class HyperworksElementCollectorBuilder : public ElementCollectorBuilder
{
public:
  HyperworksElementCollectorBuilder(void);
  ~HyperworksElementCollectorBuilder(void);

  virtual axis::application::output::collectors::ElementSetCollector& 
    BuildStressCollector( const axis::String& targetSetName, 
    axis::application::output::collectors::XXDirectionState xxState, 
    axis::application::output::collectors::YYDirectionState yyState, 
    axis::application::output::collectors::ZZDirectionState zzState, 
    axis::application::output::collectors::YZDirectionState yzState, 
    axis::application::output::collectors::XZDirectionState xzState, 
    axis::application::output::collectors::XYDirectionState xyState );

  virtual axis::application::output::collectors::ElementSetCollector& 
    BuildStrainCollector( const axis::String& targetSetName, 
    axis::application::output::collectors::XXDirectionState xxState, 
    axis::application::output::collectors::YYDirectionState yyState, 
    axis::application::output::collectors::ZZDirectionState zzState, 
    axis::application::output::collectors::YZDirectionState yzState, 
    axis::application::output::collectors::XZDirectionState xzState, 
    axis::application::output::collectors::XYDirectionState xyState );

  virtual axis::application::output::collectors::ElementSetCollector& 
    BuildPlasticStrainIncrementCollector( const axis::String& targetSetName, 
    axis::application::output::collectors::XXDirectionState xxState, 
    axis::application::output::collectors::YYDirectionState yyState, 
    axis::application::output::collectors::ZZDirectionState zzState, 
    axis::application::output::collectors::YZDirectionState yzState, 
    axis::application::output::collectors::XZDirectionState xzState, 
    axis::application::output::collectors::XYDirectionState xyState );

  virtual axis::application::output::collectors::ElementSetCollector& 
    BuildArtificialEnergyCollector( const axis::String& targetSetName );

  virtual axis::application::output::collectors::ElementSetCollector& 
    BuildEffectivePlasticStrainCollector( const axis::String& targetSetName );

  virtual axis::application::output::collectors::ElementSetCollector& 
    BuildDeformationGradientCollector( const axis::String& targetSetName );
private:
  axis::String GetFieldName(const axis::String& baseName, 
    axis::application::output::collectors::XXDirectionState xxState, 
    axis::application::output::collectors::YYDirectionState yyState, 
    axis::application::output::collectors::ZZDirectionState zzState, 
    axis::application::output::collectors::YZDirectionState yzState, 
    axis::application::output::collectors::XZDirectionState xzState, 
    axis::application::output::collectors::XYDirectionState xyState) const;
};

} } } } // namespace axis::application::factories::collectors