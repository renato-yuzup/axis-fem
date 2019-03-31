#include "DoF.hpp"
#include "Node.hpp"
#include "foundation/Axis.CommonLibrary.hpp"
#include "foundation/InvalidOperationException.hpp"
#include "domain/boundary_conditions/BoundaryCondition.hpp"
#include "System.hpp"
#include "foundation/memory/HeapBlockArena.hpp"

namespace adbc = axis::domain::boundary_conditions;
namespace ade = axis::domain::elements;
namespace afm = axis::foundation::memory;

ade::DoF::DoF(id_type id,  int localIndex, const afm::RelativePointer& node) : _parentNode(node)
{
	_id	= id;
	_localIndex = localIndex;
	_condition = NULL;
}

ade::DoF::~DoF(void)
{
	// nothing to do here
}

id_type ade::DoF::GetId( void ) const
{
	return _id;
}

adbc::BoundaryCondition& ade::DoF::GetBoundaryCondition( void ) const
{
	return *_condition;
}

void ade::DoF::SetBoundaryCondition( adbc::BoundaryCondition& condition )
{
	if (HasBoundaryConditionApplied())
	{
		if (_condition == &GetBoundaryCondition())
		{	// skip redundant call
			return;
		}

		// cannot set; should call replace instead
		throw axis::foundation::InvalidOperationException();
	}

	_condition = &condition;
}

bool ade::DoF::HasBoundaryConditionApplied( void ) const
{
	return _condition != NULL;
}

void ade::DoF::ReplaceBoundaryCondition( adbc::BoundaryCondition& condition )
{
	if (!HasBoundaryConditionApplied())
	{
		throw axis::foundation::InvalidOperationException();
	}
	_condition = &condition;
}

void ade::DoF::RemoveBoundaryCondition( void )
{
	_condition = NULL;
}

void ade::DoF::Destroy( void ) const
{
	delete this;
}

ade::Node& ade::DoF::GetParentNode( void )
{
	return absref<Node>(_parentNode);
}

const ade::Node& ade::DoF::GetParentNode( void ) const
{
	return absref<Node>(_parentNode);
}

int ade::DoF::GetLocalIndex( void ) const
{
	return _localIndex;
}

afm::RelativePointer ade::DoF::Create( id_type id, int localIndex, const afm::RelativePointer& node )
{
  afm::RelativePointer ptr = System::ModelMemory().Allocate(sizeof(DoF));
  new (*ptr) DoF(id, localIndex, node);
  return ptr;
}

#if !defined(AXIS_NO_MEMORY_ARENA)
void * ade::DoF::operator new( size_t bytes )
{
  // It is supposed that the finite element object will remain in memory
  // until the end of the program. That's why we discard the relative
  // pointer. We ignore the fact that an exception might occur in
  // constructor because if it does happen, the program will end.
  afm::RelativePointer ptr = System::ModelMemory().Allocate(bytes);
  return *ptr;
}

void ade::DoF::operator delete( void *ptr )
{
  // Since the relative pointer was discarded, we can't discard memory.
  // If it is really necessary, to free up resources, obliterating
  // memory pool is a better approach.
}

void * ade::DoF::operator new( size_t, void *ptr )
{
  return ptr;
}

void ade::DoF::operator delete( void *, void * )
{
  // Since the relative pointer was discarded, we can't discard memory.
  // If it is really necessary, to free up resources, obliterating
  // memory pool is a better approach.
}
#endif
