#pragma once
#include "HexahedronFactory.hpp"

namespace axis { namespace application { namespace factories { namespace elements {

class LinearFlanaganBelytschkoHexaFactory : public HexahedronFactory
{
public:
  LinearFlanaganBelytschkoHexaFactory(void);
  ~LinearFlanaganBelytschkoHexaFactory(void);
  static bool IsValidDefinition(
    const axis::application::parsing::core::SectionDefinition& definition);
private:
  virtual axis::domain::formulations::Formulation& BuildFormulation( 
      const axis::application::parsing::core::SectionDefinition& sectionDefinition,
      axis::domain::elements::ElementGeometry& geometry);
};

} } } } // namespace axis::application::factories::elements