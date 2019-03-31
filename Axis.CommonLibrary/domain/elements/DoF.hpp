/// <summary>
/// Contains the definition for the class axis::domain::elements::DoF.
/// </summary>
/// <author>Renato T. Yamassaki</author>

#pragma once
#include "foundation/Axis.CommonLibrary.hpp"
#include "foundation/memory/pointer.hpp"
#include "nocopy.hpp"

namespace axis { namespace domain {

namespace boundary_conditions {
// a prototype
class AXISCOMMONLIBRARY_API BoundaryCondition;
}

namespace elements {

// another prototype
class AXISCOMMONLIBRARY_API Node;

/// <summary>
/// Represents a degree-of-freedom (DoF) that a node has. In other words, one possible
/// direction of movement of a node.
/// </summary>
class AXISCOMMONLIBRARY_API DoF
{
public:
	typedef long id_type;

	/**********************************************************************************************//**
		* @fn	:::DoF(id_type id, int localIndex, Node& node);
		*
		* @brief	Creates a new degree-of-freedom.
		*
		* @author	Renato T. Yamassaki
		* @date	12 jun 2012
		*
		* @param	id				Numerical identifier which relates this
		* 							dof with its position in a global matrix.
		* @param	localIndex  	Zero-based index of the dof in the node.
		* @param [in,out]	node	The node to which this dof belongs.
		**************************************************************************************************/
	DoF(id_type id, int localIndex, const axis::foundation::memory::RelativePointer& node);

	/// <summary>
	/// Destroys this object.
	/// </summary>
	~DoF(void);

	/// <summary>
	/// Destroys this object.
	/// </summary>
	void Destroy(void) const;

	/// <summary>
	/// Returns the numerical identifier of this dof.
	/// </summary>
	id_type GetId(void) const;

	int GetLocalIndex(void) const;
	Node& GetParentNode(void);
	const Node& GetParentNode(void) const;

	/// <summary>
	/// Returns if there is boundary condition applied to this dof.
	/// </summary>
	bool HasBoundaryConditionApplied(void) const;

	/// <summary>
	/// Return the boundary condition applied to this dof.
	/// </summary>
	axis::domain::boundary_conditions::BoundaryCondition& GetBoundaryCondition(void) const;

	/// <summary>
	/// Sets a new boundary condition applied to this dof.
	/// </summary>
	/// <param name="condition">The boundary condition to be applied to this dof.</param>
	void SetBoundaryCondition(axis::domain::boundary_conditions::BoundaryCondition& condition);

	/// <summary>
	/// Sets a new boundary condition to this dof removing any applied before.
	/// </summary>
	/// <param name="condition">The boundary condition to be applied to this dof.</param>
	void ReplaceBoundaryCondition(axis::domain::boundary_conditions::BoundaryCondition& condition);

	/// <summary>
	/// Removes any boundary condition applied to this dof.
	/// </summary>
	void RemoveBoundaryCondition(void);

  static axis::foundation::memory::RelativePointer Create(
      id_type id, int localIndex, const axis::foundation::memory::RelativePointer& node);
  void *operator new(size_t bytes);
  void operator delete(void *ptr);
  void *operator new(size_t bytes, void *ptr);
  void operator delete(void *, void *);
private:
  id_type _id;
  int _localIndex;
  axis::domain::boundary_conditions::BoundaryCondition *_condition;
  axis::foundation::memory::RelativePointer _parentNode;

  DISALLOW_COPY_AND_ASSIGN(DoF);
};

} } }
