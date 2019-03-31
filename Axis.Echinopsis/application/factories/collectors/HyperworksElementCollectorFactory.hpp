#pragma once
#include "application/factories/collectors/CollectorFactory.hpp"
#include "application/factories/collectors/GeneralElementCollectorFactory.hpp"
#include "HyperworksElementCollectorBuilder.hpp"

namespace axis { namespace application { namespace factories { namespace collectors {

class HyperworksElementCollectorFactory : public CollectorFactory
{
public:
  HyperworksElementCollectorFactory(void);
  ~HyperworksElementCollectorFactory(void);

  virtual void Destroy( void ) const;

  virtual axis::services::language::parsing::ParseResult TryParse( 
      const axis::String& formatName, const axis::services::language::iterators::InputIterator& begin, 
      const axis::services::language::iterators::InputIterator& end );

  virtual CollectorBuildResult ParseAndBuild( const axis::String& formatName, 
      const axis::services::language::iterators::InputIterator& begin, 
      const axis::services::language::iterators::InputIterator& end, 
      const axis::domain::analyses::NumericalModel& model, 
      axis::application::parsing::core::ParseContext& context );
private:
  axis::application::factories::collectors::GeneralElementCollectorFactory *factory_;
  HyperworksElementCollectorBuilder builder_;
};

} } } } // namespace axis::application::factories::collectors
