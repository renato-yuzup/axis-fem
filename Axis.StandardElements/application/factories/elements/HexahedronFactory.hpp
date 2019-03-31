#pragma once
#include "domain/fwd/finite_element_fwd.hpp"
#include "domain/fwd/numerical_model.hpp"
#include "application/parsing/core/SectionDefinition.hpp"
#include "foundation/Axis.SystemBase.hpp"

namespace axis { namespace application { namespace factories { namespace elements {

class HexahedronFactory
{
public:
  HexahedronFactory(void);
  virtual ~HexahedronFactory(void);

  axis::foundation::memory::RelativePointer BuildElement(
          id_type userId, axis::domain::analyses::NumericalModel& model, id_type connectivity[],
          const axis::application::parsing::core::SectionDefinition& sectionDefinition);
  bool CanBuildElement(
          id_type userId, axis::domain::analyses::NumericalModel& model, id_type connectivity[],
          const axis::domain::materials::MaterialModel& baseMaterial,
          const axis::application::parsing::core::SectionDefinition& sectionDefinition) const;
private:
  virtual axis::domain::formulations::Formulation& BuildFormulation(
    const axis::application::parsing::core::SectionDefinition& sectionDefinition,
    axis::domain::elements::ElementGeometry& geometry) = 0;
  axis::foundation::memory::RelativePointer BuildElementGeometry(id_type connectivity[], 
    int integrPointCount, axis::domain::analyses::NumericalModel& model);
  virtual void BuildIntegrationPoint(axis::domain::elements::ElementGeometry& geometry, 
    const axis::application::parsing::core::SectionDefinition& sectionDefinition);
  virtual int GetIntegrationPointCount(
    const axis::application::parsing::core::SectionDefinition& sectionDefinition) const;
};

} } } } // namespace axis::application::factories::elements
