#pragma once
#include "HexahedronFactory.hpp"

namespace axis { namespace application { namespace factories { namespace elements {

class LinearPusoHexahedronFactory : public HexahedronFactory
{
public:
  LinearPusoHexahedronFactory(void);
  ~LinearPusoHexahedronFactory(void);
  static bool IsValidDefinition(
    const axis::application::parsing::core::SectionDefinition& definition);
private:
  virtual axis::domain::formulations::Formulation& BuildFormulation( 
    const axis::application::parsing::core::SectionDefinition& sectionDefinition,
    axis::domain::elements::ElementGeometry& geometry);
};

} } } } // namespace axis::application::factories::elements