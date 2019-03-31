/// <summary>
/// Contains definition for the class axis::domain::elements::ElementGeometry.
/// </summary>
/// <author>Renato T. Yamassaki</author>

#pragma once
#include "yuzu/common/gpu.hpp"
#include "yuzu/domain/elements/Node.hpp"
#include "yuzu/domain/integration/IntegrationPoint.hpp"
#include "yuzu/foundation/memory/RelativePointer.hpp"
#include "yuzu/foundation/blas/ColumnVector.hpp"

namespace axis { namespace yuzu { namespace domain { namespace elements {
/// <summary>
/// Represents the geometry of a finite element, which is specified by an ordered set of nodes.
/// </summary>
class ElementGeometry
{
public:
	/**********************************************************************************************//**
		* @fn	virtual ElementGeometry::~ElementGeometry(void);
		*
		* @brief	Destroys this object.
		*
		* @author	Renato T. Yamassaki
		* @date	17 abr 2012
		**************************************************************************************************/
	GPU_ONLY ~ElementGeometry(void);

	/**********************************************************************************************//**
		* @fn	void ElementGeometry::SetNode(int nodeIndex, Node& node);
		*
		* @brief	Sets a node for this geometry.
		*
		* @author	Renato T. Yamassaki
		* @date	17 abr 2012
		*
		* @param	nodeIndex   	Zero-based index of the node.
		* @param [in,out]	node	The node object to be associated with this element.
		**************************************************************************************************/
	GPU_ONLY void SetNode(int nodeIndex, const axis::yuzu::foundation::memory::RelativePointer& node);

	/**********************************************************************************************//**
		* @fn	Node& ElementGeometry::GetNode(int nodeId) const;
		*
		* @brief	Returns the geometry node allocated with the specified id.
		*
		* @author	Renato T. Yamassaki
		* @date	17 abr 2012
		*
		* @param	nodeId	The node ID.
		*
		* @return	The node.
		**************************************************************************************************/
	GPU_ONLY const Node& GetNode(int nodeId) const;

	/**********************************************************************************************//**
		* @fn	Node& ElementGeometry::GetNode(int nodeId) const;
		*
		* @brief	Returns the geometry node allocated with the specified id.
		*
		* @author	Renato T. Yamassaki
		* @date	17 abr 2012
		*
		* @param	nodeId	The node ID.
		*
		* @return	The node.
		**************************************************************************************************/
	GPU_ONLY Node& GetNode(int nodeId);

	/**********************************************************************************************//**
		* @fn	BasicNode& ElementGeometry::operator[](int nodeId) const;
		*
		* @brief	Returns the geometry node allocated with the specified id.
		*
		* @author	Renato T. Yamassaki
		* @date	17 abr 2012
		*
		* @param	nodeId	The node identifier.
		*
		* @return	The indexed value.
		**************************************************************************************************/
	GPU_ONLY const Node& operator[](int nodeId) const;

	/**********************************************************************************************//**
		* @fn	BasicNode& ElementGeometry::operator[](int nodeId) const;
		*
		* @brief	Returns the geometry node allocated with the specified id.
		*
		* @author	Renato T. Yamassaki
		* @date	17 abr 2012
		*
		* @param	nodeId	The node identifier.
		*
		* @return	The indexed value.
		**************************************************************************************************/
	GPU_ONLY Node& operator[](int nodeId);

	/// <summary>
	/// Returns the total number of nodes in this geometry.
	/// </summary>
	GPU_ONLY int GetNodeCount(void) const;
	GPU_ONLY int GetNodeIndex(const axis::yuzu::domain::elements::Node& node) const;
	GPU_ONLY int GetTotalDofCount(void) const;

  /**
    * Extracts the local quantity vector of the specified global field.
    *
    * @param [in,out] localDisplacement The local field where quantities should be written to.
    * @param globalDisplacement         The global field.
  **/
	GPU_ONLY void ExtractLocalField(axis::yuzu::foundation::blas::ColumnVector& localField, 
                                  const axis::yuzu::foundation::blas::ColumnVector& globalField) const;

	/// <summary>
	/// Returns if this geometry uses any integration point.
	/// </summary>
	GPU_ONLY bool HasIntegrationPoints(void) const;

	/// <summary>
	/// Returns an integration point.
	/// </summary>
	GPU_ONLY const axis::yuzu::domain::integration::IntegrationPoint& GetIntegrationPoint(int index) const;
  GPU_ONLY axis::yuzu::domain::integration::IntegrationPoint& GetIntegrationPoint(int index);
  GPU_ONLY void SetIntegrationPoint(int index, const axis::yuzu::foundation::memory::RelativePointer& point);
  GPU_ONLY int GetIntegrationPointCount(void) const;
	GPU_ONLY bool HasNode(const axis::yuzu::domain::elements::Node& node) const;
private:
  int numNodes_;						                               // number of nodes
  int numIntegrPoints_;                                    // number of integration points
  axis::yuzu::foundation::memory::RelativePointer nodes_;  // array of nodes
  axis::yuzu::foundation::memory::RelativePointer points_; // array of integration points

  GPU_ONLY ElementGeometry(void);
  ElementGeometry(const ElementGeometry&);
  ElementGeometry& operator =(const ElementGeometry&);
};

} } } } // namespace axis::yuzu::domain::elements
