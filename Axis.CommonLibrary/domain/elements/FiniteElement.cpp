#include "FiniteElement.hpp"
#include "domain/elements/ElementGeometry.hpp"
#include "domain/formulations/Formulation.hpp"
#include "domain/materials/MaterialModel.hpp"
#include "domain/physics/InfinitesimalState.hpp"
#include "foundation/blas/ColumnVector.hpp"
#include "foundation/blas/DenseMatrix.hpp"
#include "foundation/blas/SymmetricMatrix.hpp"
#include "foundation/blas/vector_operations.hpp"
#include "System.hpp"
#include "foundation/memory/pointer.hpp"

#include "foundation/InvalidOperationException.hpp"
#include "Foundation/BLAS/DenseMatrix.hpp"
#include "../physics/UpdatedPhysicalState.hpp"

namespace ada = axis::domain::analyses;
namespace ade = axis::domain::elements;
namespace adm = axis::domain::materials;
namespace adp = axis::domain::physics;
namespace adf = axis::domain::formulations;
namespace adi = axis::domain::integration;
namespace afb = axis::foundation::blas;
namespace afm = axis::foundation::memory;
namespace afu = axis::foundation::uuids;

ade::FiniteElement::FiniteElement( id_type id, 
  const afm::RelativePointer& geometry, adm::MaterialModel& materialModel, 
  adf::Formulation& formulation ) :
geometry_(geometry), materialModel_(&materialModel), formulation_(&formulation)
{
  physicalState_ = adp::InfinitesimalState::Create();
  internalId_ = id; externalId_ = id;
  formulation.SetElement(*this);
}

ade::FiniteElement::FiniteElement( id_type internalId, id_type userId, 
  const afm::RelativePointer& geometry, adm::MaterialModel& materialModel, 
  adf::Formulation& formulation ) :
geometry_(geometry), materialModel_(&materialModel), formulation_(&formulation)
{
  physicalState_ = adp::InfinitesimalState::Create();
  internalId_ = internalId;
  externalId_ = userId;
  formulation_->SetElement(*this);
}

ade::FiniteElement::~FiniteElement(void)
{
	formulation_->Destroy();
  absref<adp::InfinitesimalState>(physicalState_).Destroy();
  delete &geometry_;
}

void ade::FiniteElement::Destroy( void ) const
{
  delete this;
}

id_type ade::FiniteElement::GetInternalId( void ) const
{
  return internalId_;
}

id_type ade::FiniteElement::GetUserId( void ) const
{
  return externalId_;
}

ade::ElementGeometry& ade::FiniteElement::Geometry( void )
{
  return absref<ade::ElementGeometry>(geometry_);
}

const ade::ElementGeometry& ade::FiniteElement::Geometry( void ) const
{
  return absref<ade::ElementGeometry>(geometry_);
}

adm::MaterialModel& ade::FiniteElement::Material( void )
{
  return *materialModel_;
}

const adm::MaterialModel& ade::FiniteElement::Material( void ) const
{
  return *materialModel_;
}

adp::InfinitesimalState& ade::FiniteElement::PhysicalState( void )
{
  return absref<adp::InfinitesimalState>(physicalState_);
}

const adp::InfinitesimalState& ade::FiniteElement::PhysicalState( void ) const
{
  return absref<adp::InfinitesimalState>(physicalState_);
}

const afb::SymmetricMatrix& ade::FiniteElement::GetStiffness( void ) const
{
  return formulation_->GetStiffness();
}

const afb::SymmetricMatrix& ade::FiniteElement::GetConsistentMass( void ) const
{
  return formulation_->GetConsistentMass();
}

const afb::ColumnVector& ade::FiniteElement::GetLumpedMass( void ) const
{
  return formulation_->GetLumpedMass();
}

real ade::FiniteElement::GetTotalArtificialEnergy( void ) const
{
  return formulation_->GetTotalArtificialEnergy();
}

void ade::FiniteElement::AllocateMemory( void )
{
	formulation_->AllocateMemory();
}

void ade::FiniteElement::CalculateInitialState( void )
{
  absref<adp::InfinitesimalState>(physicalState_).Reset();
	formulation_->CalculateInitialState();
}

void ade::FiniteElement::ExtractLocalField(afb::ColumnVector& localField, 
  const afb::ColumnVector& globalField) const
{
  const ade::ElementGeometry& geometry = 
    absref<const ade::ElementGeometry>(geometry_);
	return geometry.ExtractLocalField(localField, globalField);
}

void ade::FiniteElement::UpdateStrain(
  const afb::ColumnVector& elementDisplacementIncrement)
{
	formulation_->UpdateStrain(elementDisplacementIncrement);
}

void ade::FiniteElement::UpdateStress(
  const afb::ColumnVector& elementDisplacementIncrement, 
  const afb::ColumnVector& elementVelocity,
  const ada::AnalysisTimeline& timeInfo)
{
  auto& geometry = absref<ade::ElementGeometry>(geometry_);
  if (geometry.HasIntegrationPoints())
  { // update stress in each integration point and then update element stress
    // as its arithmetic average
    int numPoints = geometry.GetIntegrationPointCount();
    auto& eState = absref<adp::InfinitesimalState>(physicalState_);
    auto& eStress = eState.Stress();
    auto& edStress = eState.LastStressIncrement();
    auto& plasticStrain = eState.PlasticStrain();
    real avgEffPlasticStrain = 0;
    eStress.ClearAll();
    edStress.ClearAll();
    plasticStrain.ClearAll();
    for (int pointIdx = 0; pointIdx < numPoints; ++pointIdx)
    {
      auto& p = geometry.GetIntegrationPoint(pointIdx);
      auto& state = p.State();
      adp::UpdatedPhysicalState ups(state);
      materialModel_->UpdateStresses(ups, state, timeInfo, pointIdx);
      afb::VectorSum(eStress, 1.0, eStress, 1.0, state.Stress());
      afb::VectorSum(edStress, 1.0, edStress, 1.0, state.LastStressIncrement());
      afb::VectorSum(plasticStrain, 1.0, plasticStrain, 1.0, state.PlasticStrain());
      avgEffPlasticStrain += state.EffectivePlasticStrain();
    }
    eStress.Scale(1.0 / (real)numPoints);
    edStress.Scale(1.0 / (real)numPoints);
    plasticStrain.Scale(1.0 / (real)numPoints);
    avgEffPlasticStrain /= (real)numPoints;
    eState.EffectivePlasticStrain() = avgEffPlasticStrain;
  }
  else
  { 
    auto& state = absref<adp::InfinitesimalState>(physicalState_);
    adp::UpdatedPhysicalState ups(state);
    materialModel_->UpdateStresses(ups, state, timeInfo, 0);
  }
}

void ade::FiniteElement::UpdateInternalForce( afb::ColumnVector& internalForce, 
  const afb::ColumnVector& elementDisplacementIncrement,
  const afb::ColumnVector& elementVelocity, 
  const ada::AnalysisTimeline& timeInfo)
{
  formulation_->UpdateInternalForce(internalForce, 
    elementDisplacementIncrement, elementVelocity, timeInfo);
}

void axis::domain::elements::FiniteElement::UpdateGeometry( void )
{
  formulation_->UpdateGeometry();
}

void ade::FiniteElement::UpdateMatrices( const MatrixOption& whichMatrices, 
  const afb::ColumnVector& elementDisplacement, 
  const afb::ColumnVector& elementVelocity)
{
	formulation_->UpdateMatrices(whichMatrices, elementDisplacement, 
    elementVelocity);
}

void ade::FiniteElement::ClearMemory( void )
{
	formulation_->ClearMemory();
}

real ade::FiniteElement::GetCriticalTimestep( 
  const afb::ColumnVector& elementDisplacement ) const
{
	return formulation_->GetCriticalTimestep(elementDisplacement);
}

bool ade::FiniteElement::IsCPUCapable( void ) const
{
  return formulation_->IsCPUCapable() && materialModel_->IsCPUCapable();
}

bool ade::FiniteElement::IsGPUCapable( void ) const
{
  return formulation_->IsGPUCapable() && materialModel_->IsGPUCapable();
}

afu::Uuid ade::FiniteElement::GetFormulationTypeId( void ) const
{
  return formulation_->GetTypeId();
}

afu::Uuid ade::FiniteElement::GetMaterialTypeId( void ) const
{
  return materialModel_->GetTypeId();
}

size_type ade::FiniteElement::GetFormulationBlockSize( void ) const
{
  const ElementGeometry& g = absref<ElementGeometry>(geometry_);
  int ndof = g.GetTotalDofCount();
  size_type formulationBlockSize = ndof*ndof * sizeof(real); // matrix output
  formulationBlockSize += sizeof(real); // artificial energy
  formulationBlockSize += formulation_->GetGPUDataLength(); // specific data
  return formulationBlockSize;
}

size_type ade::FiniteElement::GetMaterialBlockSize( void ) const
{
  size_type materialBlockSize = sizeof(real) * 4; // general properties
  materialBlockSize += 36 * sizeof(real); // material tensor
  materialBlockSize += materialModel_->GetDataBlockSize(); // specific data
  return materialBlockSize;
}

void ade::FiniteElement::InitializeGPUFormulation( void *baseDataAddress )
{
  const ElementGeometry& g = absref<ElementGeometry>(geometry_);
  int ndof = g.GetTotalDofCount();
  real *artificialEnergySlot = 
    (real *)((uint64)baseDataAddress + (ndof*ndof * sizeof(real)));
  void *formulationSpecificSlot = &artificialEnergySlot[1];
  formulation_->InitializeGPUData(formulationSpecificSlot, 
    artificialEnergySlot);
}

void ade::FiniteElement::InitializeGPUMaterial( void *baseDataAddress )
{
  uint64 baseAddr            = (uint64)baseDataAddress;
  real *densitySlot          = (real *)baseDataAddress;
  real *waveSpeedSlot        = &densitySlot[1];
  real *bulkModulusSlot      = &densitySlot[2];
  real *shearModulusSlot     = &densitySlot[3];
  real *tensorSlot           = &densitySlot[4];
  void *materialSpecificSlot = &tensorSlot[36];
  materialModel_->InitializeGPUData(materialSpecificSlot, densitySlot, 
    waveSpeedSlot, bulkModulusSlot, shearModulusSlot, tensorSlot);
}

afm::RelativePointer ade::FiniteElement::Create( id_type id, 
  const afm::RelativePointer& geometry, adm::MaterialModel& materialModel, 
  adf::Formulation& formulation )
{
  return Create(id, id, geometry, materialModel, formulation);
}

afm::RelativePointer ade::FiniteElement::Create( id_type internalId, 
  id_type userId, const afm::RelativePointer& geometry, 
  adm::MaterialModel& materialModel, adf::Formulation& formulation )
{
  auto ptr = System::ModelMemory().Allocate(sizeof(FiniteElement));
  new (*ptr) FiniteElement(internalId, userId, geometry, materialModel, 
    formulation);
  return ptr;
}

void * ade::FiniteElement::operator new( size_t bytes )
{
  auto ptr = System::ModelMemory().Allocate(bytes);
  return *ptr;
}

void * ade::FiniteElement::operator new( size_t, void *ptr )
{
  return ptr;
}

void ade::FiniteElement::operator delete( void * )
{
  // nothing to do here
}

void ade::FiniteElement::operator delete( void *, void * )
{
  // nothing to do here
}

adf::FormulationStrategy& ade::FiniteElement::GetGPUFormulationStrategy( void )
{
  return formulation_->GetGPUStrategy();
}

adm::MaterialStrategy& ade::FiniteElement::GetGPUMaterialStrategy( void )
{
  return materialModel_->GetGPUStrategy();
}
