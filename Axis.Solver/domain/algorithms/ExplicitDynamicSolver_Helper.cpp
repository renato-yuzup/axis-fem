#include "ExplicitDynamicSolver_Helper.hpp"
#include "domain/elements/ElementGeometry.hpp"
#include "domain/elements/FiniteElement.hpp"
#include "domain/analyses/NumericalModel.hpp"
#include "domain/elements/MatrixOption.hpp"
#include "foundation/blas/VectorView.hpp"
#include "foundation/memory/pointer.hpp"
#include "dof_status.hpp"

namespace ada = axis::domain::analyses;
namespace adb = axis::domain::boundary_conditions;
namespace adc = axis::domain::collections;
namespace ade = axis::domain::elements;
namespace afb = axis::foundation::blas;
namespace afm = axis::foundation::memory;

void InitializeModelVectors( ada::NumericalModel& model )
{
  ada::ModelDynamics& dynamics = model.Dynamics();
  ada::ModelKinematics& kinematics = model.Kinematics();
  size_type totalNumDofs = model.GetTotalDofCount();

  if (!dynamics.IsReactionForceAvailable()) 
  {
    dynamics.ResetReactionForce(totalNumDofs);
  }
  if (!dynamics.IsInternalForceAvailable()) 
  {
    dynamics.ResetInternalForce(totalNumDofs);
  }
  if (!dynamics.IsExternalLoadAvailable())
  {
    dynamics.ResetExternalLoad(totalNumDofs);
  }
  else
  {
    dynamics.ExternalLoads().ClearAll();
  }
  if (!kinematics.IsAccelerationFieldAvailable()) 
    kinematics.ResetAcceleration(totalNumDofs);
  if (!kinematics.IsVelocityFieldAvailable()) 
    kinematics.ResetVelocity(totalNumDofs);
  if (!kinematics.IsDisplacementFieldAvailable()) 
    kinematics.ResetDisplacement(totalNumDofs);
  if (!kinematics.IsDisplacementIncrementFieldAvailable()) 
    kinematics.ResetDisplacementIncrement(totalNumDofs);
}

void InitializeVectorMask(afm::RelativePointer& maskPtr,
  const ada::NumericalModel& model)
{
  uint64 totalDofCount = model.GetTotalDofCount();
  size_type nodeCount = model.Nodes().Count();
  maskPtr = axis::System::ModelMemory().Allocate(totalDofCount * sizeof(char));
  char *mask = axis::absptr<char>(maskPtr);

  // initialize dof status
  for (size_type nodeIdx = 0; nodeIdx < nodeCount; ++nodeIdx)
  {
    ade::Node& node = model.Nodes().GetByInternalIndex(nodeIdx);
    int dofCount = node.GetDofCount();
    for (int dofIdx = 0; dofIdx < dofCount; dofIdx++)
    {
      ade::DoF& dof = node[dofIdx];
      size_type id = dof.GetId();
      char status = DOF_STATUS_FREE;
      if (dof.HasBoundaryConditionApplied())	// a lock or prescription
      {
        adb::BoundaryCondition& bc = dof.GetBoundaryCondition();
        if (bc.IsLock())
        {
          status = DOF_STATUS_LOCKED;
        }
        else if (bc.IsPrescribedVelocity())
        {
          status = DOF_STATUS_PRESCRIBED_VELOCITY;
        }
        else if (bc.IsPrescribedDisplacement())
        {
          status = DOF_STATUS_PRESCRIBED_DISPLACEMENT;
        }
      }
      mask[id] = status;
    }
  }
}

void CalculateExplicitInitialCondition( ada::NumericalModel& model, 
  const ada::AnalysisTimeline& timeline )
{
  ada::ModelDynamics& dynamics = model.Dynamics();
  // calculate reaction force
  afb::VectorSum(dynamics.ReactionForce(), -1.0, dynamics.ExternalLoads(), 
    -1.0, dynamics.InternalForces());
}

void UpdateNodeCoordinates( ada::NumericalModel& model, 
  const afb::ColumnVector& displacementIncrement)
{
  auto& nodes = model.Nodes();
  size_type nodeCount = nodes.Count();
  #pragma omp parallel for
  for (size_type nodeIdx = 0; nodeIdx < nodeCount; ++nodeIdx)
  {
    ade::Node& node = nodes.GetByInternalIndex(nodeIdx);
    id_type dofXIdx = node.GetDoF(0).GetId();
    id_type dofYIdx = node.GetDoF(1).GetId();
    id_type dofZIdx = node.GetDoF(2).GetId();
    node.CurrentX() += displacementIncrement(dofXIdx);
    node.CurrentY() += displacementIncrement(dofYIdx);
    node.CurrentZ() += displacementIncrement(dofZIdx);
  }
}

void CalculateElementMassMatrices( ada::NumericalModel& model )
{
  // calculate mass matrix of all elements
  size_type count = model.Elements().Count();
  ade::LumpedMassOnlyOption matrixOpt;
  ada::ModelKinematics& kinematics = model.Kinematics();
  afb::ColumnVector& modelDisplacement = kinematics.Displacement();
  afb::ColumnVector& modelVelocity = kinematics.Velocity();

  #pragma omp parallel for SOLVER_OMP_SCHEDULE_DEFAULT
  for (size_type idx = 0; idx < count; ++idx)
  {
    ade::FiniteElement& element = model.Elements().GetByInternalIndex(idx);
    element.UpdateMatrices(matrixOpt, modelDisplacement, modelVelocity);
  }
}

void BuildGlobalMassMatrix( afm::RelativePointer& globalLumpedMassPtr, 
  ada::NumericalModel& model )
{
  auto& mass = axis::absref<afb::ColumnVector>(globalLumpedMassPtr);
  adc::NodeSet& nodes = model.Nodes();
  size_type nodeCount = nodes.Count();
  mass.ClearAll();
  CalculateElementMassMatrices(model);

  #pragma omp parallel for SOLVER_OMP_SCHEDULE_DEFAULT
  for (size_type nodeIdx = 0; nodeIdx < nodeCount; nodeIdx++)
  {
    ade::Node& node = nodes.GetByInternalIndex(nodeIdx);
    int numDofs = node.GetDofCount();
    int numElem = node.GetConnectedElementCount();
    for (int elemIdx = 0; elemIdx < numElem; ++elemIdx)
    {
      const ade::FiniteElement& element = node.GetConnectedElement(elemIdx);
      const afb::ColumnVector& lumpedMass = element.GetLumpedMass();
      int nodeLocalId = element.Geometry().GetNodeIndex(node);
      for (int dofIdx = 0; dofIdx < numDofs; ++dofIdx)
      {
        const ade::DoF& dof = node[dofIdx];
        size_type globalId = dof.GetId();
        mass(globalId) += lumpedMass(numDofs * nodeLocalId + dofIdx);
      }
    }
  }
}

void ApplyPrescribedDisplacements( afb::ColumnVector& displacementIncrement, 
  const afb::ColumnVector& currentTotalDisplacement, 
  const adc::DofList& appliedConditions, real time, char *vectorMask )
{
  size_type count = appliedConditions.Count();
  #pragma omp parallel for schedule(dynamic, 256)
  for (size_type idx = 0; idx < count; ++idx)
  {
    const ade::DoF& dof = appliedConditions[idx];
    const adb::BoundaryCondition& bc = dof.GetBoundaryCondition();
    size_type dofIdx = dof.GetId();

    if (bc.Active(time))
    {
      // update displacement increment
      real newDisplacement = bc.GetValue(time);
      displacementIncrement(dofIdx) = newDisplacement - 
        currentTotalDisplacement(dofIdx);		
    }
    else
    {
      vectorMask[dofIdx] = DOF_STATUS_FREE;
    }
  }
}

void ApplyBoundaryConditionsToField( afb::ColumnVector& field, 
  const adc::DofList& bcList, real time, char *vectorMask )
{
  size_type count = bcList.Count();
  #pragma omp parallel for schedule(dynamic, 256)
  for (size_type idx = 0; idx < count; ++idx)
  {
    const ade::DoF& dof = bcList[idx];
    const adb::BoundaryCondition& bc = dof.GetBoundaryCondition();
    uint64 dofId = dof.GetId();
    if (bc.Active(time))
    {
      real val = bc.GetValue(time);
      field(dofId) = val;
    }
    else
    {
      vectorMask[dofId] = DOF_STATUS_FREE;
    }
  }
}

void ApplyPrescribedVelocities( afb::ColumnVector& velocity, afb::ColumnVector& acceleration, 
								const adc::DofList& bcList, real time, real dt, char *vectorMask )
{
	size_type count = bcList.Count();
	#pragma omp parallel for schedule(dynamic, 256)
	for (size_type idx = 0; idx < count; ++idx)
	{
		const ade::DoF& dof = bcList[idx];
		const adb::BoundaryCondition& bc = dof.GetBoundaryCondition();
    id_type id = dof.GetId();
    if (bc.Active(time))
    {
      real val = bc.GetValue(time);
      real dv = val - velocity(id);
      velocity(id) = val;
      if (dt != 0)
      {
        acceleration(id) = dv / dt;
      }
    }
    else
    {
      if (vectorMask[id] != DOF_STATUS_FREE)
      {
        vectorMask[id] = DOF_STATUS_FREE;
        // keep velocity as it is
        acceleration(id) = 0;
      }
    }
	}
}

void UpdateBoundaryConditions( ada::NumericalModel& model, real time, real dt, char *vectorMask )
{
  auto& locks = model.Locks();
  auto& externalLoads = model.Dynamics().ExternalLoads();
  auto& nodalLoads = model.NodalLoads();
  auto& velocity = model.Kinematics().Velocity();
  auto& acceleration = model.Kinematics().Acceleration();
  auto& appliedVelocities = model.AppliedVelocities();
  auto& displacement = model.Kinematics().Displacement();
  auto& dispIncrement = model.Kinematics().DisplacementIncrement();
  auto& appliedDisplacements = model.AppliedDisplacements();
  ApplyBoundaryConditionsToField(dispIncrement, locks, time, vectorMask);
  ApplyBoundaryConditionsToField(externalLoads, nodalLoads, time, vectorMask);

  ApplyPrescribedVelocities(velocity, acceleration, appliedVelocities, time, dt, vectorMask);
  ApplyPrescribedDisplacements(dispIncrement, displacement, 
    appliedDisplacements, time, vectorMask);
}

void UpdateModelStressState( ada::NumericalModel& model, 
  const afb::ColumnVector& displacementIncrement,
  const afb::ColumnVector& velocity, const ada::AnalysisTimeline& timeline )
{
  adc::ElementSet& elements = model.Elements();
  size_type count = elements.Count();

  #pragma omp parallel for SOLVER_OMP_SCHEDULE_DEFAULT
  for (size_type i = 0; i < count; ++i)
  {
    ade::FiniteElement& element = elements.GetByInternalIndex(i);
    int totalDof = element.Geometry().GetTotalDofCount();
    afb::ColumnVector localDisplacement(totalDof), localVelocity(totalDof);
    element.ExtractLocalField(localDisplacement, displacementIncrement);
    element.ExtractLocalField(localVelocity, velocity);
    element.UpdateStrain(localDisplacement);
    element.UpdateStress(localDisplacement, localVelocity, timeline);
  }
}

void CalculateInternalForce( afb::ColumnVector& internalForce, 
  const ada::NumericalModel& model, const afb::ColumnVector& displacement,
  const afb::ColumnVector& displacementIncrement, 
  const afb::ColumnVector& velocity, const ada::AnalysisTimeline& timeline)
{
  adc::ElementSet& elements = model.Elements();
  size_type count = elements.Count();
  internalForce.ClearAll();

  #pragma omp parallel for SOLVER_OMP_SCHEDULE_DEFAULT
  for (size_type i = 0; i < count; ++i)
  {
    ade::FiniteElement& element = elements.GetByInternalIndex(i);
    ade::ElementGeometry& g = element.Geometry();
    int dofCount = g.GetTotalDofCount();
    int nodeCount = g.GetNodeCount();
    int dofPerNode = g.GetNode(0).GetDofCount();
    afb::ColumnVector elementInternalForces(dofCount);
    afb::ColumnVector locDisplacIncrement(dofCount), locVelocity(dofCount);
    element.ExtractLocalField(locDisplacIncrement, displacementIncrement);
    element.ExtractLocalField(locVelocity, velocity);
    element.UpdateInternalForce(elementInternalForces, locDisplacIncrement, 
      locVelocity, timeline);

    #pragma omp critical (_axis_explicit_standard_solver_internal_force_region_)
    {
      for (int nodeIdx = 0; nodeIdx < nodeCount; nodeIdx++)
      {
        ade::Node& node = g.GetNode(nodeIdx);
        for (int dofIdx = 0; dofIdx < dofPerNode; dofIdx++)
        {
          size_type dofId = node[dofIdx].GetId();
          internalForce(dofId) += 
            elementInternalForces(nodeIdx*dofPerNode + dofIdx);
        }
      }
    }
  }
}

void UpdateGeometryInformation( ada::NumericalModel& model )
{
  adc::ElementSet& elements = model.Elements();
  size_type count = elements.Count();

  #pragma omp parallel for SOLVER_OMP_SCHEDULE_DEFAULT
  for (size_type i = 0; i < count; ++i)
  {
    ade::FiniteElement& element = elements.GetByInternalIndex(i);
    element.UpdateGeometry();
  }
}
