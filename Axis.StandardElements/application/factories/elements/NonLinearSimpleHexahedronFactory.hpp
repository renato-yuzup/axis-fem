#pragma once
#include "HexahedronFactory.hpp"

namespace axis { namespace application { namespace factories { namespace elements {

class NonLinearSimpleHexahedronFactory : public HexahedronFactory
{
public:
  NonLinearSimpleHexahedronFactory(void);
  ~NonLinearSimpleHexahedronFactory(void);
  static bool IsValidDefinition(
    const axis::application::parsing::core::SectionDefinition& definition);
private:
  virtual axis::domain::formulations::Formulation& BuildFormulation( 
    const axis::application::parsing::core::SectionDefinition& sectionDefinition,
    axis::domain::elements::ElementGeometry& geometry);
  virtual void BuildIntegrationPoint( axis::domain::elements::ElementGeometry& geometry, 
    const axis::application::parsing::core::SectionDefinition& sectionDefinition );
  virtual int GetIntegrationPointCount( 
    const axis::application::parsing::core::SectionDefinition& sectionDefinition ) const;
};

} } } } // namespace axis::application::factories::elements