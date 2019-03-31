/// <summary>
/// Contains definition for the class axis::domain::elements::ElementGeometry.
/// </summary>
/// <author>Renato T. Yamassaki</author>

#pragma once
#include "foundation/Axis.CommonLibrary.hpp"
#include "foundation/blas/blas.hpp"
#include "domain/elements/Node.hpp"
#include "domain/integration/IntegrationPoint.hpp"

namespace axis { namespace domain { namespace elements {
/// <summary>
/// Represents the geometry of a finite element, which is specified by an ordered set of nodes.
/// </summary>
class AXISCOMMONLIBRARY_API ElementGeometry
{
public:

	/**********************************************************************************************//**
		* @fn	ElementGeometry::ElementGeometry(int numNodes, int numFaces, int numEdges,
		* 		axis::domain::integration::IntegrationDimension& integrationPoints);
		*
		* @brief	Creates a new instance of this class.
		*
		* @author	Renato T. Yamassaki
		* @date	17 abr 2012
		*
		* @param	numNodes				 	The number of nodes contained
		* 										in this geometry.
		* @param	numFaces				 	The number of faces contained
		* 										in this geometry.
		* @param	numEdges				 	The number of edges contained
		* 										in this geometry.
		* @param [in,out]	integrationPoints	Container of integration
		* 										points in this geometry.
		**************************************************************************************************/
	ElementGeometry(int numNodes, int integrationPointCount);

	/**********************************************************************************************//**
		* @fn	ElementGeometry::ElementGeometry(int numNodes, int numFaces, int numEdges);
		*
		* @brief	Creates a new instance of this class.
		*
		* @author	Renato T. Yamassaki
		* @date	17 abr 2012
		*
		* @param	numNodes	The number of nodes contained in this
		* 						geometry.
		* @param	numFaces	The number of faces contained in this
		* 						geometry.
		* @param	numEdges	The number of edges contained in this
		* 						geometry.
		**************************************************************************************************/
	ElementGeometry(int numNodes);

	/**********************************************************************************************//**
		* @fn	virtual ElementGeometry::~ElementGeometry(void);
		*
		* @brief	Destroys this object.
		*
		* @author	Renato T. Yamassaki
		* @date	17 abr 2012
		**************************************************************************************************/
	~ElementGeometry(void);

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
	void SetNode(int nodeIndex, const axis::foundation::memory::RelativePointer& node);

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
	const Node& GetNode(int nodeId) const;

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
	Node& GetNode(int nodeId);

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
	const Node& operator[](int nodeId) const;

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
	Node& operator[](int nodeId);

	/// <summary>
	/// Returns the total number of nodes in this geometry.
	/// </summary>
	int GetNodeCount(void) const;

	int GetNodeIndex(const axis::domain::elements::Node& node) const;

	int GetTotalDofCount(void) const;

  /**
    * Extracts the local quantity vector of the specified global field.
    *
    * @param [in,out] localDisplacement The local field where quantities should be written to.
    * @param globalDisplacement         The global field.
  **/
	void ExtractLocalField(axis::foundation::blas::ColumnVector& localField, 
                         const axis::foundation::blas::ColumnVector& globalField) const;

	/// <summary>
	/// Returns if this geometry uses any integration point.
	/// </summary>
	bool HasIntegrationPoints(void) const;

	/// <summary>
	/// Returns an integration point.
	/// </summary>
	const axis::domain::integration::IntegrationPoint& GetIntegrationPoint(int index) const;
  axis::domain::integration::IntegrationPoint& GetIntegrationPoint(int index);

  void SetIntegrationPoint(int index, const axis::foundation::memory::RelativePointer& point);

  int GetIntegrationPointCount(void) const;

	bool HasNode(const axis::domain::elements::Node& node) const;

  static axis::foundation::memory::RelativePointer Create(int numNodes, int integrationPointCount);
  static axis::foundation::memory::RelativePointer Create(int numNodes);

  void *operator new(size_t bytes);
  void *operator new(size_t bytes, void *ptr);
  void operator delete(void *ptr);
  void operator delete(void *, void *);
private:
  int numNodes_;						                        // number of nodes
  int numIntegrPoints_;                             // number of integration points
  axis::foundation::memory::RelativePointer nodes_; // array of nodes
  axis::foundation::memory::RelativePointer points_;// array of integration points

  void InitGeometry(int numNodes);
};

} } } // namespace axis::domain::elements
