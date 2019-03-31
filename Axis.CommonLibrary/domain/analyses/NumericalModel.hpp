#pragma once

#include "foundation/Axis.CommonLibrary.hpp"
#include "foundation/Axis.SystemBase.hpp"
#include "AxisString.hpp"

#include "domain/collections/NodeSet.hpp"
#include "domain/collections/ElementSet.hpp"
#include "domain/collections/NodeSetCollection.hpp"
#include "domain/collections/ElementSetCollection.hpp"
#include "domain/collections/CurveSet.hpp"
#include "domain/collections/BoundaryConditionList.hpp"
#include "domain/collections/DofList.hpp"
#include "foundation/date_time/Timestamp.hpp"
#include "foundation/uuids/Uuid.hpp"
#include "ModelKinematics.hpp"
#include "ModelDynamics.hpp"

namespace axis { namespace domain { namespace analyses {

class AXISCOMMONLIBRARY_API NumericalModel
{
private:
	// Stores and manages identification of objects
	id_type _nextNodeId;
	id_type _nextElementId;
	id_type _nextDofId;

	// Our collections
	axis::domain::collections::NodeSet *_nodes;
	axis::domain::collections::ElementSet *_elements;
	axis::domain::collections::CurveSet _curves;

	axis::domain::collections::DofList& _boundaryConditions;

	axis::domain::collections::DofList& _appliedNodalLoads;
	axis::domain::collections::DofList& _lockConstraints;
	axis::domain::collections::DofList& _appliedDisplacements;
	axis::domain::collections::DofList& _appliedAccelerations;
	axis::domain::collections::DofList& _appliedVelocities;
				
	axis::foundation::memory::RelativePointer _kinematics;
	axis::foundation::memory::RelativePointer _dynamics;

	axis::domain::collections::NodeSetCollection& _nodeSets;
	axis::domain::collections::ElementSetCollection& _elementSets;

	NumericalModel(void);
public:
	virtual ~NumericalModel(void);

	static axis::domain::analyses::NumericalModel& Create(void);

	axis::domain::collections::NodeSet& Nodes(void) const;
	axis::domain::collections::NodeSet& GetNodeSet(const axis::String& alias) const;
	void AddNodeSet(const axis::String& alias, axis::domain::collections::NodeSet& nodeSet);
	void RemoveNodeSet(const axis::String& alias);
	bool ExistsNodeSet(const axis::String& alias) const;

	axis::domain::collections::ElementSet& Elements(void) const;
	axis::domain::collections::ElementSet& GetElementSet(const axis::String& id) const;
	void AddElementSet(const axis::String& id, axis::domain::collections::ElementSet& elementSet);
	void RemoveElementSet(const axis::String& id);
	bool ExistsElementSet(const axis::String& id) const;

	axis::domain::collections::CurveSet& Curves(void);
	const axis::domain::collections::CurveSet& Curves(void) const;

	const axis::domain::collections::DofList& AllBoundaryConditions(void) const;
	axis::domain::collections::DofList& AllBoundaryConditions(void);

	axis::domain::collections::DofList& NodalLoads(void);
	const axis::domain::collections::DofList& NodalLoads(void) const;

	const axis::domain::collections::DofList& Locks(void) const;
	axis::domain::collections::DofList& Locks(void);

	const axis::domain::collections::DofList& AppliedDisplacements(void) const;
	axis::domain::collections::DofList& AppliedDisplacements(void);

	const axis::domain::collections::DofList& AppliedAccelerations(void) const;
	axis::domain::collections::DofList& AppliedAccelerations(void);

	const axis::domain::collections::DofList& AppliedVelocities(void) const;
	axis::domain::collections::DofList& AppliedVelocities(void);

  bool IsGPUCapable(void) const;

	id_type PeekNextNodeId(void) const;
	id_type PopNextNodeId(void);

	id_type PeekNextElementId(void) const;
	id_type PopNextElementId(void);

	id_type PeekNextDofId(void) const;
	id_type PopNextDofId(void);
	id_type PopNextDofId(int dofCount);

	/**********************************************************************************************//**
		* <summary> Returns the total number of degrees of freedom in the model.</summary>
		*
		* <returns> The total degrees of freedom count.</returns>
		**************************************************************************************************/
	size_type GetTotalDofCount(void) const;

	axis::domain::analyses::ModelKinematics& Kinematics(void);
	const axis::domain::analyses::ModelKinematics& Kinematics(void) const;
  axis::foundation::memory::RelativePointer GetKinematicsPointer(void);

	axis::domain::analyses::ModelDynamics& Dynamics(void);
	const axis::domain::analyses::ModelDynamics& Dynamics(void) const;
  axis::foundation::memory::RelativePointer GetDynamicsPointer(void);

	void ResetMesh( void );

  void InitStep(void);
};		

} } }
