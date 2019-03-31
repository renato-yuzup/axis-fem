#include "common_omp.hpp"
#include "NumericalModel.hpp"
#include "foundation/ArgumentException.hpp"
#include "foundation/InvalidOperationException.hpp"
#include "foundation/memory/pointer.hpp"
#include "domain/collections/NodeSet.hpp"
#include "domain/collections/ElementSet.hpp"
#include "ModelDynamics.hpp"
#include "ModelKinematics.hpp"

namespace ada = axis::domain::analyses;
namespace adc = axis::domain::collections;
namespace ade = axis::domain::elements;
namespace afm = axis::foundation::memory;

ada::NumericalModel::NumericalModel(void) : 
_nodeSets(*new adc::NodeSetCollection()), _elementSets(*new adc::ElementSetCollection()), 
_appliedNodalLoads(*new adc::DofList()),
_lockConstraints(*new adc::DofList()),
_appliedDisplacements(*new adc::DofList()),
_appliedVelocities(*new adc::DofList()),
_appliedAccelerations(*new adc::DofList()),
_boundaryConditions(*new adc::DofList())
{
	_nodes = new adc::NodeSet();
	_elements = new adc::ElementSet();

  _kinematics = ModelKinematics::Create();
	_dynamics = ModelDynamics::Create();

	_nextNodeId = 0;
	_nextElementId = 0;
	_nextDofId = 0;
}

ada::NumericalModel::~NumericalModel(void)
{
	// Destroy every element and nodes contained herein.
	// First, destroy all elements...
	_elements->DestroyAll();

	// ...then, destroy all nodes...
	_nodes->DestroyAll();

	// and finally, destroy all containers (sets)
	_nodeSets.DestroyChildren();
  _elementSets.DestroyChildren();
	_nodes->Destroy();
	_elements->Destroy();
	_boundaryConditions.Destroy();
	_lockConstraints.Destroy();
	_appliedDisplacements.Destroy();
	_appliedAccelerations.Destroy();
	_appliedVelocities.Destroy();
	if (_kinematics != NULLPTR) System::ModelMemory().Deallocate(_kinematics);
	if (_dynamics   != NULLPTR) System::ModelMemory().Deallocate(_dynamics);

  _kinematics = NULLPTR;
  _dynamics   = NULLPTR;
	_nodes      = NULL;
	_elements   = NULL;
}

adc::NodeSet& ada::NumericalModel::Nodes( void ) const
{
	return *_nodes;
}

bool ada::NumericalModel::ExistsNodeSet( const axis::String& alias ) const
{
	return _nodeSets.Contains(alias);
}

adc::NodeSet& ada::NumericalModel::GetNodeSet( const axis::String& alias ) const
{
	if (!ExistsNodeSet(alias))
	{
		throw axis::foundation::ArgumentException();
	}
	return static_cast<adc::NodeSet&>(_nodeSets[alias]);
}

void ada::NumericalModel::AddNodeSet( const axis::String& alias, adc::NodeSet& nodeSet )
{
	if (ExistsNodeSet(alias))
	{
		throw axis::foundation::ArgumentException();
	}	
	_nodeSets.Add(alias, nodeSet);
}

void ada::NumericalModel::RemoveNodeSet( const axis::String& alias )
{
	if (!ExistsNodeSet(alias))
	{
		throw axis::foundation::ArgumentException();
	}
	_nodeSets.Remove(alias);
}

adc::ElementSet& ada::NumericalModel::Elements( void ) const
{
	return *_elements;
}

adc::ElementSet& ada::NumericalModel::GetElementSet( const axis::String& id ) const
{
	if (!_elementSets.Contains(id))
	{
		throw axis::foundation::ArgumentException();
	}
	return static_cast<adc::ElementSet&>(_elementSets[id]);
}

void ada::NumericalModel::AddElementSet( const axis::String& id, adc::ElementSet& elementSet )
{
	if (ExistsElementSet(id))
	{
		throw axis::foundation::ArgumentException();
	}
	_elementSets.Add(id, elementSet);
}

void ada::NumericalModel::RemoveElementSet( const axis::String& id )
{
	if (!ExistsElementSet(id))
	{
		throw axis::foundation::ArgumentException();
	}
	_elementSets.Remove(id);
}

bool ada::NumericalModel::ExistsElementSet( const axis::String& id ) const
{
	return _elementSets.Contains(id);
}

ada::NumericalModel& ada::NumericalModel::Create( void )
{
	return *new NumericalModel();
}

adc::CurveSet& ada::NumericalModel::Curves( void )
{
	return _curves;
}

const adc::CurveSet& ada::NumericalModel::Curves( void ) const
{
	return _curves;
}

id_type ada::NumericalModel::PopNextNodeId( void )
{
	return _nextNodeId++;	// return and increment
}

adc::DofList& ada::NumericalModel::NodalLoads( void )
{
	return _appliedNodalLoads;
}

const adc::DofList& ada::NumericalModel::NodalLoads( void ) const
{
	return _appliedNodalLoads;
}

id_type ada::NumericalModel::PeekNextNodeId( void ) const
{
	return _nextNodeId;
}

id_type ada::NumericalModel::PeekNextElementId( void ) const
{
	return _nextElementId;
}

id_type ada::NumericalModel::PopNextElementId( void )
{
	return _nextElementId++;	// return and increment
}

id_type ada::NumericalModel::PeekNextDofId( void ) const
{
	return _nextDofId;
}

id_type ada::NumericalModel::PopNextDofId( void )
{
	return _nextDofId++;
}

id_type ada::NumericalModel::PopNextDofId( int dofCount )
{
	if (dofCount <= 0)
	{
		throw axis::foundation::ArgumentException(_T("dofCount"));
	}
	id_type nextDof = _nextDofId;
	_nextDofId += dofCount;
	return nextDof;
}

ada::ModelKinematics& ada::NumericalModel::Kinematics( void )
{
  return absref<ModelKinematics>(_kinematics);
}

const ada::ModelKinematics& ada::NumericalModel::Kinematics( void ) const
{
  return absref<ModelKinematics>(_kinematics);
}

ada::ModelDynamics& ada::NumericalModel::Dynamics( void )
{
  return absref<ModelDynamics>(_dynamics);
}

const ada::ModelDynamics& ada::NumericalModel::Dynamics( void ) const
{
	return absref<ModelDynamics>(_dynamics);
}

void ada::NumericalModel::ResetMesh( void )
{
	adc::ElementSet& elements = Elements();
	size_type count = Elements().Count();
	
	#pragma omp parallel for COMMON_SCHEDULE_SMALL_OPS
	for (size_type i = 0; i < count; ++i)
	{
		axis::domain::elements::FiniteElement& element = elements.GetByPosition(i);
		element.AllocateMemory();
	}
}

void ada::NumericalModel::InitStep( void )
{
  adc::ElementSet& elements = Elements();
  size_type count = Elements().Count();

  #pragma omp parallel for COMMON_SCHEDULE_SMALL_OPS
  for (size_type i = 0; i < count; ++i)
  {
    axis::domain::elements::FiniteElement& element = elements.GetByPosition(i);
    element.CalculateInitialState();
  }
}

const adc::DofList& ada::NumericalModel::Locks( void ) const
{
	return _lockConstraints;
}

adc::DofList& ada::NumericalModel::Locks( void )
{
	return _lockConstraints;
}

const adc::DofList& ada::NumericalModel::AllBoundaryConditions( void ) const
{
	return _boundaryConditions;
}

adc::DofList& ada::NumericalModel::AllBoundaryConditions( void )
{
	return _boundaryConditions;
}

const adc::DofList& ada::NumericalModel::AppliedDisplacements( void ) const
{
	return _appliedDisplacements;
}

adc::DofList& ada::NumericalModel::AppliedDisplacements( void )
{
	return _appliedDisplacements;
}

const adc::DofList& ada::NumericalModel::AppliedAccelerations( void ) const
{
	return _appliedAccelerations;
}

adc::DofList& ada::NumericalModel::AppliedAccelerations( void )
{
	return _appliedAccelerations;
}

const adc::DofList& ada::NumericalModel::AppliedVelocities( void ) const
{
	return _appliedVelocities;
}

adc::DofList& ada::NumericalModel::AppliedVelocities( void )
{
	return _appliedVelocities;
}

size_type ada::NumericalModel::GetTotalDofCount( void ) const
{
	return _nextDofId;
}

afm::RelativePointer ada::NumericalModel::GetKinematicsPointer( void )
{
  return _kinematics;
}

afm::RelativePointer ada::NumericalModel::GetDynamicsPointer( void )
{
  return _dynamics;
}

bool ada::NumericalModel::IsGPUCapable( void ) const
{
  bool isCapable = true;
  const adc::ElementSet& elements = Elements();
  size_type elementCount = elements.Count();
  for (size_type i = 0; i < elementCount && isCapable; ++i)
  {
    const axis::domain::elements::FiniteElement& element = elements.GetByPosition(i);
    isCapable = element.IsGPUCapable();
  }
  
  const adc::CurveSet& curves = Curves();
  size_type curveCount = curves.Count();
  for (size_type i = 0; i < curveCount && isCapable; ++i)
  {
    const axis::domain::curves::Curve& c = curves[i];
    isCapable = c.IsGPUCapable();
  }

  const adc::DofList& bcList = AllBoundaryConditions();
  size_type bcCount = bcList.Count();
  for (size_type i = 0; i < bcCount && isCapable; ++i)
  {
    const axis::domain::elements::DoF& dof = bcList[i];
    const axis::domain::boundary_conditions::BoundaryCondition& bc = dof.GetBoundaryCondition();
    isCapable = bc.IsGPUCapable();
  }
  
  return isCapable;
}
