#include "stdafx.h"
#include "HexahedronFactory.hpp"
#include "domain/elements/ElementGeometry.hpp"
#include "domain/analyses/NumericalModel.hpp"
#include "foundation/memory/pointer.hpp"

namespace aapc = axis::application::parsing::core;
namespace aafe = axis::application::factories::elements;
namespace ada = axis::domain::analyses;
namespace adc = axis::domain::collections;
namespace ade = axis::domain::elements;
namespace adi = axis::domain::integration;
namespace adf = axis::domain::formulations;
namespace adm = axis::domain::materials;
namespace afm = axis::foundation::memory;

aafe::HexahedronFactory::HexahedronFactory( void )
{
  // nothing to do here
}

aafe::HexahedronFactory::~HexahedronFactory( void )
{
  // nothing to do here
}

afm::RelativePointer aafe::HexahedronFactory::BuildElement( id_type userId, 
  ada::NumericalModel& model, id_type connectivity[], 
  const aapc::SectionDefinition& sectionDefinition )
{
  int integrPointCount = GetIntegrationPointCount(sectionDefinition);

  // 1) Builg basic element geometry associations.
  afm::RelativePointer ptrGeom = BuildElementGeometry(connectivity, 
    integrPointCount, model);
  ade::ElementGeometry& geometry = absref<ade::ElementGeometry>(ptrGeom);

  // 2) If required, build and associate integration points to this geometry.
  BuildIntegrationPoint(geometry, sectionDefinition);

  // 3) Build corresponding element formulation.
  auto& formulation = BuildFormulation(sectionDefinition, geometry);

  // 4) Clone material model according to number of integration points in the
  // geometry.
  auto& material = sectionDefinition.GetMaterial().Clone(integrPointCount);

  // 5) Glue everything together to build a finite element
  id_type internalId = model.PopNextElementId();
  afm::RelativePointer ptr = ade::FiniteElement::Create(internalId, userId, 
    ptrGeom, material, formulation);
  ade::FiniteElement& element = absref<ade::FiniteElement>(ptr);

  // 6) Update node connectivity (reverse connectivity)
  adc::NodeSet& nodes = model.Nodes();
  for (int i = 0; i < 8; i++)
  {
    id_type nodeId = connectivity[i];
    ade::Node& node = nodes.GetByUserIndex(nodeId);
    if (!node.WasInitialized())
    {
      id_type nextDofId = model.PopNextDofId(3);
      node.InitDofs(3, nextDofId);
    }
    node.ConnectElement(ptr);
  }
  return ptr;
}

axis::foundation::memory::RelativePointer aafe::HexahedronFactory::BuildElementGeometry( id_type connectivity[], 
    int integrPointCount, ada::NumericalModel& model )
{
  afm::RelativePointer ptr = ade::ElementGeometry::Create(8, integrPointCount);
  ade::ElementGeometry& geometry = absref<ade::ElementGeometry>(ptr);
  for (int i = 0; i < 8; i++)
  {
    id_type nodeId = connectivity[i];
    afm::RelativePointer nodePtr = model.Nodes().GetPointerByUserId(nodeId);
    geometry.SetNode(i, nodePtr);
  }
  return ptr;
}

void aafe::HexahedronFactory::BuildIntegrationPoint( ade::ElementGeometry& geometry, 
                                                     const aapc::SectionDefinition& sectionDefinition )
{
  // nothing to do in base implementation
}

int aafe::HexahedronFactory::GetIntegrationPointCount( const aapc::SectionDefinition& sectionDefinition ) const
{
  return 0;
}