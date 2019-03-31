/// <summary>
/// Contains the definition for the class axis::domain::elements::DoF.
/// </summary>
/// <author>Renato T. Yamassaki</author>

#pragma once
#include "yuzu/common/gpu.hpp"
#include "yuzu/foundation/memory/RelativePointer.hpp"

namespace axis { namespace yuzu { namespace domain {

namespace elements {

// another prototype
class Node;

/// <summary>
/// Represents a degree-of-freedom (DoF) that a node has. In other words, one possible
/// direction of movement of a node.
/// </summary>
class DoF
{
public:
	/// <summary>
	/// Destroys this object.
	/// </summary>
	GPU_ONLY ~DoF(void);

	/// <summary>
	/// Returns the numerical identifier of this dof.
	/// </summary>
	GPU_ONLY id_type GetId(void) const;

	GPU_ONLY int GetLocalIndex(void) const;
	GPU_ONLY Node& GetParentNode(void);
	GPU_ONLY const Node& GetParentNode(void) const;

	/// <summary>
	/// Returns if there is boundary condition applied to this dof.
	/// </summary>
	GPU_ONLY bool HasBoundaryConditionApplied(void) const;
private:
  id_type _id;
  int _localIndex;
  void *_condition;
  axis::yuzu::foundation::memory::RelativePointer _parentNode;

  // disallow creating a new dof
  GPU_ONLY DoF(void);
  DoF(const DoF&);
  DoF& operator =(const DoF&);
};

} } } } // namespace axis::yuzu::domain::elements
