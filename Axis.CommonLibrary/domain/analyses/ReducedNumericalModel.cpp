#include "ReducedNumericalModel.hpp"
#include "ModelOperatorFacade.hpp"
#include "foundation/memory/pointer.hpp"
#include "NumericalModel.hpp"

namespace ada = axis::domain::analyses;
namespace adc = axis::domain::collections;
namespace ade = axis::domain::elements;
namespace afm = axis::foundation::memory;

ada::ReducedNumericalModel::ReducedNumericalModel( NumericalModel& sourceModel, 
                                                   ModelOperatorFacade& op ) :
kinematics_(sourceModel.GetKinematicsPointer()), dynamics_(sourceModel.GetDynamicsPointer()),
operator_(&op)
{
  elementCount_ = sourceModel.Elements().Count();
  nodeCount_ = sourceModel.Nodes().Count();
  elementArrayPtr_ = System::ModelMemory().Allocate(sizeof(afm::RelativePointer) * elementCount_);
  nodeArrayPtr_ = System::ModelMemory().Allocate(sizeof(afm::RelativePointer) * nodeCount_);
  outputBucketArrayPtr_ = System::ModelMemory().Allocate(sizeof(void *) * elementCount_);

  adc::ElementSet& elements = sourceModel.Elements();
  for (size_type eIdx = 0; eIdx < elementCount_; eIdx++)
  {
    afm::RelativePointer& ptr = absptr<afm::RelativePointer>(elementArrayPtr_)[eIdx];
    ptr = elements.GetPointerByInternalId(eIdx);
  }

  adc::NodeSet& nodes = sourceModel.Nodes();
  for (size_type nIdx = 0; nIdx < nodeCount_; nIdx++)
  {
    afm::RelativePointer& ptr = absptr<afm::RelativePointer>(nodeArrayPtr_)[nIdx];
    ptr = nodes.GetPointerByInternalId(nIdx);
  }
}

ada::ReducedNumericalModel::~ReducedNumericalModel( void )
{
  operator_->Destroy();
}

afm::RelativePointer ada::ReducedNumericalModel::Create( NumericalModel& sourceModel, 
                                                         ModelOperatorFacade& op )
{
  afm::RelativePointer ptr = System::ModelMemory().Allocate(sizeof(ReducedNumericalModel));
  new (*ptr) ReducedNumericalModel(sourceModel, op);
  return ptr;
}

void * ada::ReducedNumericalModel::operator new( size_t, void *ptr )
{
  return ptr;
}

void ada::ReducedNumericalModel::operator delete( void *, void * )
{
  // nothing to do here
}

const ada::ModelDynamics& ada::ReducedNumericalModel::Dynamics( void ) const
{
  return absref<ada::ModelDynamics>(dynamics_);
}

ada::ModelDynamics& ada::ReducedNumericalModel::Dynamics( void )
{
  return absref<ada::ModelDynamics>(dynamics_);
}

const ada::ModelKinematics& ada::ReducedNumericalModel::Kinematics( void ) const
{
  return absref<ada::ModelKinematics>(kinematics_);
}

ada::ModelKinematics& ada::ReducedNumericalModel::Kinematics( void )
{
  return absref<ada::ModelKinematics>(kinematics_);
}

size_type ada::ReducedNumericalModel::GetElementCount( void ) const
{
  return elementCount_;
}

size_type ada::ReducedNumericalModel::GetNodeCount( void ) const
{
  return nodeCount_;
}

const ade::FiniteElement& ada::ReducedNumericalModel::GetElement( size_type index ) const
{
  return absref<ade::FiniteElement>(absptr<afm::RelativePointer>(elementArrayPtr_)[index]);
}

ade::FiniteElement& ada::ReducedNumericalModel::GetElement( size_type index )
{
  return absref<ade::FiniteElement>(absptr<afm::RelativePointer>(elementArrayPtr_)[index]);
}

const afm::RelativePointer ada::ReducedNumericalModel::GetElementPointer( size_type index ) const
{
  return absptr<afm::RelativePointer>(elementArrayPtr_)[index];
}

afm::RelativePointer ada::ReducedNumericalModel::GetElementPointer( size_type index )
{
  return absptr<afm::RelativePointer>(elementArrayPtr_)[index];
}

const ade::Node& ada::ReducedNumericalModel::GetNode( size_type index ) const
{
  return absref<ade::Node>(absptr<afm::RelativePointer>(nodeArrayPtr_)[index]);
}

ade::Node& ada::ReducedNumericalModel::GetNode( size_type index )
{
  return absref<ade::Node>(absptr<afm::RelativePointer>(nodeArrayPtr_)[index]);
}

const afm::RelativePointer ada::ReducedNumericalModel::GetNodePointer( size_type index ) const
{
  return absptr<afm::RelativePointer>(nodeArrayPtr_)[index];
}

afm::RelativePointer ada::ReducedNumericalModel::GetNodePointer( size_type index )
{
  return absptr<afm::RelativePointer>(nodeArrayPtr_)[index];
}

ada::ModelOperatorFacade& ada::ReducedNumericalModel::GetOperator( void )
{
  return *operator_;
}
