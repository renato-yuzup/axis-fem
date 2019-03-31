#include "CPULinearConjugateGradientSolver.hpp"
#include <math.h>
#include "domain/algorithms/messages/ModelStateUpdateMessage.hpp"
#include "domain/elements/FiniteElement.hpp"
#include "domain/elements/ElementGeometry.hpp"
#include "domain/elements/MatrixOption.hpp"
#include "domain/physics/InfinitesimalState.hpp"
#include "services/diagnostics/Process.hpp"
#include "services/locales/LocaleLocator.hpp"
#include "services/messaging/LogMessage.hpp"
#include "foundation/date_time/Timestamp.hpp"
#include "foundation/InvalidOperationException.hpp"
#if !defined(AXIS_NO_MEMORY_ARENA)
#include "foundation/memory/pointer.hpp"
#endif

namespace ada = axis::domain::analyses;
namespace adam = axis::domain::algorithms::messages;
namespace adal = axis::domain::algorithms;
namespace adb = axis::domain::boundary_conditions;
namespace adc = axis::domain::collections;
namespace ade = axis::domain::elements;
namespace adp = axis::domain::physics;
namespace asdi = axis::services::diagnostics::information;
namespace aslc = axis::services::locales;
namespace afb = axis::foundation::blas;
namespace afd = axis::foundation::date_time;
namespace afm = axis::foundation::memory;
namespace asmm = axis::services::messaging;
namespace asd = axis::services::diagnostics;

/*--------------------------------------- AUXILIARY FUNCTIONS ------------------------------------------------*/
axis::String _FormatDataSize(storage_space_type amount)
{
	storage_space_type sz = amount;
	axis::String unit = _T(" bytes");
	if (sz >= 1000)
	{
		sz /= 1000;
		unit = _T(" kB");
	}
	if (sz >= 1000)
	{
		sz /= 1000;
		unit = _T(" MB");
	}
	return axis::String::int_parse(sz) + unit;
}
/*------------------------------------------------------------------------------------------------------------*/

adal::CPULinearConjugateGradientSolver::CPULinearConjugateGradientSolver( Clockwork& clock ) : 
ConjugateGradientSolver(clock)
{
	_elementaryStiffnessCalculated = false;
	_Qn = NULLPTR;
	_R0 = NULL;
	_U0 = NULL;
	_analysis = NULL;
}

adal::CPULinearConjugateGradientSolver::~CPULinearConjugateGradientSolver( void )
{
	// nothing to do here
}

void adal::CPULinearConjugateGradientSolver::AssembleExternalLoadVector( afb::ColumnVector& externalLoadVector )
{
	adc::DofList& loadList = _analysis->NodalLoads();
	long loadCount = _analysis->NodalLoads().Count();

	#pragma omp parallel for shared(externalLoadVector, loadList) SOLVER_OMP_SCHEDULE_DEFAULT
	for (size_type i = 0; i < loadCount; ++i)
	{
		// actually, every element of the array is a nodal load, so
		// this cast is safe
		id_type vectorPos = loadList[i].GetId();

		// since only one BC acts on a dof, it's safe not to
		// use mutex
		externalLoadVector(vectorPos) = static_cast<adb::BoundaryCondition&>(
                                        loadList[i].GetBoundaryCondition()).GetValue(0);
	}

	// now, apply loads equivalent to prescribed displacements that
	// have been set
	if (!_analysis->AppliedDisplacements().Empty())
	{	// we have some nodes with prescribed displacement; we need 
		// to apply their contributions to the load vector
		CalculateElementsStiffnessMatrix();
		ApplyPrescribedDisplacementsToLoadVector(externalLoadVector);
	}
}

void adal::CPULinearConjugateGradientSolver::ApplyPrescribedDisplacementsToLoadVector( afb::ColumnVector& externalLoadVector )
{
	adc::DofList& appliedDisplacements = _analysis->AppliedDisplacements();
	size_type count = appliedDisplacements.Count();
	for (size_type i = 0; i < count; ++i)	// for each prescribed displacement
	{
		const ade::DoF& constrainedDof = appliedDisplacements[i];
		const ade::Node& constrainedNode = constrainedDof.GetParentNode();
		const adb::BoundaryCondition& constraint = static_cast<adb::BoundaryCondition&>(
                                                    constrainedDof.GetBoundaryCondition());
		const real displacementValue = constraint.GetValue(0);
		const int constrainedDofLocalIdx = constrainedDof.GetLocalIndex();
		
		int constrainedElementsCount = constrainedNode.GetConnectedElementCount();
		for (int elemIdx = 0; elemIdx < constrainedElementsCount; ++elemIdx)	// for each element affected by the prescribed displacement
		{
			const ade::FiniteElement& fe = constrainedNode.GetConnectedElement(elemIdx);
			const ade::ElementGeometry& geometry = fe.Geometry();
			const afb::SymmetricMatrix& stiffness = fe.GetStiffness();
			int constrainedNodeLocalIdx = geometry.GetNodeIndex(constrainedNode);

			int dofPerNode = geometry.GetNode(0).GetDofCount();	// we suppose every node in the element has the same dof count
			int elemNodeCount = geometry.GetNodeCount();
			#pragma omp parallel for shared(elemNodeCount, dofPerNode, geometry, stiffness, constrainedDofLocalIdx, displacementValue) SOLVER_OMP_SCHEDULE_DEFAULT
			for (int nodeIdx = 0; nodeIdx < elemNodeCount; ++nodeIdx)	// for each node in the affected element
			{
				const ade::Node& affectedNode = geometry.GetNode(nodeIdx);

				#pragma omp parallel for shared(elemNodeCount, dofPerNode, geometry, stiffness, constrainedDofLocalIdx, displacementValue, affectedNode) SOLVER_OMP_SCHEDULE_DEFAULT
				for (int dofIdx = 0; dofIdx < dofPerNode; ++dofIdx)
				{
					const ade::DoF& affectedDof = affectedNode.GetDoF(dofIdx);
					bool isConstrained = false;
					if (affectedDof.HasBoundaryConditionApplied())
					{
						const adb::BoundaryCondition& bc = affectedDof.GetBoundaryCondition();
						isConstrained = bc.IsPrescribedDisplacement(); // bc.IsVariableConstraint();	// ignore if other constraint is already in action in this dof
					}
					if (&affectedDof != &constrainedDof && !isConstrained)
					{
						const real kCoeff = stiffness.GetElement(nodeIdx*dofPerNode + dofIdx, constrainedNodeLocalIdx*dofPerNode + constrainedDofLocalIdx);
						real reaction = kCoeff * displacementValue;
						externalLoadVector(affectedDof.GetId()) = externalLoadVector(affectedDof.GetId()) - reaction;
					}
				}
			}
		}
	}		
}

afb::ColumnVector& adal::CPULinearConjugateGradientSolver::AssembleQ( const afb::ColumnVector& searchDirectionVector )
{
  afb::ColumnVector& Qn = absref<afb::ColumnVector>(_Qn);
  size_type count = _analysis->Nodes().Count();
  adc::NodeSet& nodes = _analysis->Nodes();
	for (size_type i = 0; i < count; ++i)
	{
		ade::Node& node = nodes.GetByPosition(i);
		int dofCount = node.GetDofCount();
		for (int dofIndex = 0; dofIndex < dofCount; ++dofIndex)
		{
			real q = CalculateQCoefficient(node, searchDirectionVector, dofIndex);
			Qn(node.GetDoF(dofIndex).GetId()) = q;
		}
	}
	return Qn;
}

real adal::CPULinearConjugateGradientSolver::CalculateQCoefficient( const ade::Node& node, 
                                                                    const afb::ColumnVector& P, 
                                                                    int dofIndex ) const
{
	// if this node has a movement constraint, just return the value of P
	if (IsDofMovementConstrained(node, dofIndex))
	{
		return 0; // P.GetElement(node.GetDoF(dofIndex).GetId(), 0);
	}
	real q = 0;
	size_type elementCount = node.GetConnectedElementCount();

 	#pragma omp parallel for shared(elementCount, node) reduction(+:q) SOLVER_OMP_SCHEDULE_DEFAULT
	for (id_type i = 0; i < elementCount; ++i)
	{
		ade::FiniteElement& element = node.GetConnectedElement(i);
		ade::ElementGeometry& g = element.Geometry();
		const afb::SymmetricMatrix& ke = element.GetStiffness();
		int elemNodeCount = g.GetNodeCount();
		int localRowNodeIndex = g.GetNodeIndex(node);
		int localRowPos = localRowNodeIndex*3 + dofIndex;

		// iterate through a row of the stiffness matrix
		real kp = 0;
 		#pragma omp parallel for shared(elemNodeCount, localRowNodeIndex, dofIndex) reduction(+:kp) SOLVER_OMP_SCHEDULE_DEFAULT
		for (int j = 0; j < elemNodeCount; ++j)
		{
			ade::Node& elementNode = g.GetNode(j);
			int dofCount = elementNode.GetDofCount();
 			#pragma omp parallel for shared(localRowNodeIndex, dofIndex) reduction(+:kp) SOLVER_OMP_SCHEDULE_DEFAULT
			for (int n = 0; n < dofCount; ++n)	// iterate through dof's
			{
				// if dof is movement constrained, it doesn't contribute to
				// global stiffness
				if (IsDofMovementConstrained(elementNode, n))
				{
					kp += 0;	// is it necessary to OpenMP?
				}
				else
				{
					size_type globalColumnPos = elementNode.GetDoF(n).GetId();
					size_type localColumnPos = j*3 + n;
					real k = ke.GetElement(localRowPos, localColumnPos);
					real p = P(globalColumnPos);
					real parcel = (k*p);

					if (IsDebugEnabled())
					{
						String s = _T("ke[") + String::int_parse(i) + _T("](");
						s += String::int_parse((long)localRowPos) + _T(",") + String::int_parse(localColumnPos) + _T(") * ");
						s += _T("P(") + String::int_parse(globalColumnPos) + _T(") = ");
						s += String::double_parse(parcel);
						LogDebugMessage(s);
					}
					kp += parcel;
				}
			}
		}
		q += kp;
	}
	if (this->IsDebugEnabled())
	{
		String s = _T("Q(") + String::int_parse(node.GetDoF(dofIndex).GetId()) + _T(") = ");
		s += String::double_parse(q);
		LogDebugMessage(s);
	}
	return q;
}

void adal::CPULinearConjugateGradientSolver::CalculateElementsStiffnessMatrix( void )
{
	if (_elementaryStiffnessCalculated) return;		// do not calculate again

  afb::ColumnVector& modelDisplacement = _analysis->Kinematics().Displacement();
  afb::ColumnVector& modelVelocity = _analysis->Kinematics().Velocity();
	id_type elemCount = _analysis->Elements().Count();

	#pragma omp parallel for shared(elemCount) SOLVER_OMP_SCHEDULE_DEFAULT
	for (id_type i = 0; i < elemCount; ++i)
	{
		ade::FiniteElement& e = _analysis->Elements().GetByInternalIndex(i);
		e.CalculateInitialState();
		e.UpdateMatrices(ade::StiffnessMatrixOnlyOption(), modelDisplacement, modelVelocity);
	}

	_elementaryStiffnessCalculated = true;
}

void adal::CPULinearConjugateGradientSolver::ExecutePostProcessing( const afb::ColumnVector& solutionVector, 
                                                                    ada::NumericalModel& model, 
                                                                    long iterationCount, 
                                                                    const ada::AnalysisTimeline& timeInfo )
{
	const aslc::Locale& loc = aslc::LocaleLocator::GetLocator().GetGlobalLocale();

	// Now, calculate stress and strain for all elements and nodes
	afd::Timespan cgRunTime = afd::Timestamp::GetUTCTime() - _algorithmStartTime;
	CalculateElementsStressStrain(solutionVector, timeInfo);
	CalculateNodesStressStrain();
	afd::Timespan totalRunTime = afd::Timestamp::GetUTCTime() - _algorithmStartTime;

	// Do post-processing
	DispatchMessage(asmm::LogMessage(_T("Triggering post-processing...")));
	afd::Timestamp ppStart = afd::Timestamp::GetUTCTime();
  DispatchMessage(adam::ModelStateUpdateMessage(model.Kinematics(), model.Dynamics()));
	afd::Timestamp ppEnd = afd::Timestamp::GetUTCTime();
	afd::Timespan ppDuration = ppEnd - ppStart;
	DispatchMessage(asmm::LogMessage(_T("Post-processing ok. Elapsed time: ") + loc.GetDataTimeLocale().ToShortTimeRangeString(ppDuration)));
	
	// Print summary
	asd::Process process = asd::Process::GetCurrentProcess();
	uint64 maxUsedPhysMemorySize = process.GetPeakPhysicalMemoryAllocated();
	uint64 maxUsedVirtualMemorySize = process.GetPeakPhysicalMemoryAllocated();
  DispatchMessage(asmm::LogMessage(_T("***")));
  DispatchMessage(asmm::LogMessage(_T("LSCG SOLVER SUMMARY")));
  DispatchMessage(asmm::LogMessage(asmm::LogMessage::NestingOpen));
  DispatchMessage(asmm::LogMessage(_T("Total CG run time        : ") + loc.GetDataTimeLocale().ToShortTimeRangeString(cgRunTime)));
  DispatchMessage(asmm::LogMessage(_T("Total algorithm run time : ") + loc.GetDataTimeLocale().ToShortTimeRangeString(totalRunTime)));
  DispatchMessage(asmm::LogMessage(_T("Iteration count          : ") + String::int_parse(iterationCount)));
  DispatchMessage(asmm::LogMessage(asmm::LogMessage::NestingClose));
}

void adal::CPULinearConjugateGradientSolver::ExecuteInitialSteps( ada::NumericalModel& analysis )
{
	/*
		On algorithm startup, three things must be done:
		   1) Initialize internal variables
		   2) Allocate memory for algorithm vectors
		   3) Calculate stiffness matrix of every element in the model
	*/

	// use given analysis object
	_analysis = &analysis;
	long nodeCount = _analysis->Nodes().Count();

	// start timing
	_algorithmStartTime = afd::Timestamp::GetUTCTime();

	// allocate space for the displacement and nodal loads
	if (!analysis.Kinematics().IsDisplacementFieldAvailable())
	{
		analysis.Kinematics().ResetDisplacement(nodeCount*3);
	}
  if (!analysis.Kinematics().IsVelocityFieldAvailable())
  {
    analysis.Kinematics().ResetVelocity(nodeCount*3);
  }
	if (!analysis.Dynamics().IsExternalLoadAvailable())
	{
		analysis.Dynamics().ResetExternalLoad(nodeCount*3);
	}

	LogSolverInfoMessage(0, _T("Processing phase 1 of memory allocation."));
#if defined(AXIS_NO_MEMORY_ARENA)
	_Qn = new afb::ColumnVector(analysis.GetTotalDofCount());
#else
  _Qn = afb::ColumnVector::CreateFromGlobalMemory(analysis.GetTotalDofCount());
#endif
	_R0 = &analysis.Dynamics().ExternalLoads();
	_U0 = &analysis.Kinematics().Displacement();
	absref<afb::ColumnVector>(_Qn).ClearAll();	

	LogSolverInfoMessage(0, _T("Calculating stiffness of the elements."));
	CalculateElementsStiffnessMatrix();
}

void adal::CPULinearConjugateGradientSolver::ExecuteCleanupSteps( ada::NumericalModel& analysis )
{
	// free resources
	absref<afb::ColumnVector>(_Qn).Destroy();
#if !defined(AXIS_NO_MEMORY_ARENA)
  System::GlobalMemory().Deallocate(_Qn);
#endif
	_Qn = NULLPTR;
	_U0 = NULL;
	_R0 = NULL;
}

void adal::CPULinearConjugateGradientSolver::WarnLongWait( long currentIterationStep )
{
	String msg = _T("Still looking for convergence. Entering iteration ");
	msg += String::int_parse(currentIterationStep);
	msg += _T(".");
	LogSolverInfoMessage(0, msg);
}

void adal::CPULinearConjugateGradientSolver::AbortSolutionProcedure( long lastIterationStep )
{
	// warn user
	LogSolverWarningMessage(1, _T("The solver took too much time to compute an accurate solution. The last solution will be used as an approximation."));
}

afb::ColumnVector& adal::CPULinearConjugateGradientSolver::GetInitializedResidualWorkVector( void )
{
	// the initial residual vector is the effective load vector,
	// that is, {R} = {Fext} - [K]{U}, but {U} = {0}.
	LogSolverInfoMessage(0, _T("Evaluating external loads..."));
// 	_R0->ClearAll();
	AssembleExternalLoadVector(*_R0);
	return *_R0;
}

afb::ColumnVector& adal::CPULinearConjugateGradientSolver::GetInitializedSolutionWorkVector( void )
{
	// the first guess for the solution vector is a null vector
	// with only the prescribed displacements set
// 	_U0->ClearAll();

	adc::DofList& appliedDisplacements = _analysis->AppliedDisplacements();
	size_type count = appliedDisplacements.Count();
	for (size_type i = 0; i < count; ++i)
	{
		ade::DoF& dof = appliedDisplacements[i];
		adb::BoundaryCondition& constraint = static_cast<adb::BoundaryCondition&>(dof.GetBoundaryCondition());
		_U0->SetElement(dof.GetId(), constraint.GetValue(0));
	}

	return *_U0;
}

long adal::CPULinearConjugateGradientSolver::GetMaximumIterationsAllowed( void ) const
{
	return 20000;
}

long adal::CPULinearConjugateGradientSolver::GetNumStepsToLongComputation( void ) const
{
	return 200;
}

bool adal::CPULinearConjugateGradientSolver::IsDofMovementConstrained( const ade::Node& node, int dofIndex ) const
{
	if (node.GetDoF(dofIndex).HasBoundaryConditionApplied())
	{
		if (node.GetDoF(dofIndex).GetBoundaryCondition().IsLock())
		{
			return true;
		}
	}
	return false;
}

int adal::CPULinearConjugateGradientSolver::GetSolverEventSourceId( void ) const
{
	return 0x071;
}

void adal::CPULinearConjugateGradientSolver::LogSectionHeader( const axis::String& sectionName ) const
{
	const char_type pattern = '*';
	int headerWidth = 128;
	int semiWidth;
	if (sectionName.size() >= headerWidth - 2)
	{
		semiWidth = 2;
		headerWidth = (int)sectionName.size() + 6;
	}
	else
	{
		semiWidth = (headerWidth - (int)sectionName.size() - 2) / 2;
	}
	axis::String separatorLine(headerWidth, pattern);
	axis::String headerLine =	axis::String(semiWidth, pattern) + _T(" ") +
											sectionName + _T(" ") +
											axis::String(semiWidth, pattern);
  this->DispatchMessage(asmm::LogMessage(separatorLine));
  this->DispatchMessage(asmm::LogMessage(headerLine));
}

void adal::CPULinearConjugateGradientSolver::LogSectionFooter( const axis::String& sectionName ) const
{
	const char_type pattern = '*';
	int headerWidth = 128;
	int semiWidth;
	if (sectionName.size() >= headerWidth - 2)
	{
		semiWidth = 2;
		headerWidth = (int)sectionName.size() + 6;
	}
	else
	{
		semiWidth = (headerWidth - (int)sectionName.size() - 2) / 2;
	}
	axis::String separatorLine(headerWidth, pattern);
	axis::String footerLine =	axis::String(semiWidth, pattern) + _T(" ") +
											sectionName + _T(" ") +
											axis::String(semiWidth, pattern);
  this->DispatchMessage(asmm::LogMessage(footerLine));
  this->DispatchMessage(asmm::LogMessage(separatorLine));
}

void adal::CPULinearConjugateGradientSolver::CalculateElementsStressStrain( 
  const afb::ColumnVector& solutionVector, const ada::AnalysisTimeline& timeInfo)
{
  afb::ColumnVector& velocity = _analysis->Kinematics().Velocity();
	adc::ElementSet& elements = _analysis->Elements();
	size_type elementCount = _analysis->Elements().Count();
	
	#pragma omp parallel for shared(elementCount, elements) SOLVER_OMP_SCHEDULE_DEFAULT
	for (size_type i = 0; i < elementCount; ++i)
	{
		ade::FiniteElement& e = elements.GetByPosition(i);
		int vectorSize = e.Geometry().GetTotalDofCount();
		afb::ColumnVector localDisplacement(vectorSize), localVelocity(vectorSize);
		
		// obtain local displacements
		e.ExtractLocalField(localDisplacement, solutionVector);
    e.ExtractLocalField(localVelocity, velocity);

		// calculate strain and then stress
		e.UpdateStrain(localDisplacement);
		e.UpdateStress(localDisplacement, localVelocity, timeInfo);
	}
}

void adal::CPULinearConjugateGradientSolver::CalculateNodesStressStrain( void )
{
	adc::NodeSet& nodes = _analysis->Nodes();
	size_type nodeCount = _analysis->Nodes().Count();

	#pragma omp parallel for shared(nodeCount, nodes) SOLVER_OMP_SCHEDULE_DEFAULT
	for (size_type i = 0; i < nodeCount; ++i)
	{
		ade::Node& node = nodes.GetByPosition(i);
		node.ResetStrain();
		node.ResetStress();

		int elementCount = node.GetConnectedElementCount();
		for (int elementIdx = 0; elementIdx < elementCount; ++elementIdx)
		{
			ade::FiniteElement& e = node.GetConnectedElement(elementIdx);
      adp::InfinitesimalState& state = e.PhysicalState();
      afb::VectorSum(node.Strain(), node.Strain(), 1.0, state.Strain());
      afb::VectorSum(node.Stress(), node.Stress(), 1.0, state.Stress());
		}

		node.Strain().Scale(1.0 / (real)elementCount);
		node.Stress().Scale(1.0 / (real)elementCount);
	}
}

real adal::CPULinearConjugateGradientSolver::CalculateRhsVectorNorm( void )
{
	size_type count = _R0->Rows();
	real norm = 0;

	// we sum with initial displacement guess due to prescribed
	// displacements
	#pragma omp parallel for shared(count) reduction(+:norm) SOLVER_OMP_SCHEDULE_DEFAULT
	for (size_type i = 0; i < count; ++i)
	{
		real parcel = _R0->GetElement(i);
		parcel += _U0->GetElement(i);
		parcel = parcel * parcel;
		norm += parcel;
	}
	norm = sqrt(norm);

	if (norm == 0.0)	// this is almost impossible, but just in case...
	{
		norm = ErrorTolerance;
	}

	return norm;
}

real adal::CPULinearConjugateGradientSolver::CalculateRhsScalarProduct( const afb::ColumnVector& rightFactor )
{
	size_type count = _R0->Rows();
	real product = 0;

	// we sum with initial displacement guess due to prescribed
	// displacements
	#pragma omp parallel for shared(count) reduction(+:product) SOLVER_OMP_SCHEDULE_DEFAULT
	for (size_type i = 0; i < count; ++i)
	{
		real parcel = _R0->GetElement(i);
		parcel += _U0->GetElement(i);
		parcel = parcel * rightFactor.GetElement(i);
		product += parcel;
	}

	return product;
}

void adal::CPULinearConjugateGradientSolver::Destroy( void ) const
{
	delete this;
}

asdi::SolverCapabilities adal::CPULinearConjugateGradientSolver::GetCapabilities( void ) const
{
	return asdi::SolverCapabilities(_T("LSCG"),
	                                _T("A conjugate gradient-based solver for linear static problems."), 
	                                false, false, false, false);
}