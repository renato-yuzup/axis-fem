/// <summary>
/// Contains definition for the class axis::domain::elements::Node.
/// </summary>
/// <author>Renato T. Yamassaki</author>
#pragma once

#include "foundation/Axis.CommonLibrary.hpp"
#include "foundation/blas/blas.hpp"
#include "foundation/collections/ObjectList.hpp"
#include "domain/collections/ElementSet.hpp"
#include "domain/elements/DoF.hpp"

#include "foundation/ElementNotFoundException.hpp"
#include "foundation/OutOfBoundsException.hpp"

namespace axis { namespace domain { namespace elements {

class AXISCOMMONLIBRARY_API FiniteElement;

/// <summary>
/// Represents a node used to connect and define finite elements.
/// </summary>
class AXISCOMMONLIBRARY_API Node
{
public:
	typedef long id_type;

	/**********************************************************************************************//**
		* @fn	:::Node(id_type id);
		*
		* @brief	Creates a new node with the number of dof's given.
		*
		* @author	Renato T. Yamassaki
		* @date	05 abr 2012
		*
		* @param	id	Numerical identifier which relates this node with its
		* 				position in a global matrix.
		*
		* ### param	numDofs	Number of dof's.
		**************************************************************************************************/
	Node(id_type id);

	/**********************************************************************************************//**
		* @fn	:::Node(id_type internalId, id_type externalId);
		*
		* @brief	Creates a new node with the number of dof's given.
		*
		* @author	Renato T. Yamassaki
		* @date	05 abr 2012
		*
		* @param	internalId	Numerical identifier which relates this node
		* 						with its position in a global matrix.
		* @param	externalId	External numerical identifier.
		*
		* ### param	numDofs	Number of dof's.
		**************************************************************************************************/
	Node(id_type internalId, id_type externalId);

	/**********************************************************************************************//**
		* @fn	:::Node(id_type id, coordtype x, coordtype y, coordtype z);
		*
		* @brief	Creates a new node with the number of dof's given.
		*
		* @author	Renato T. Yamassaki
		* @date	05 abr 2012
		*
		* @param	id	Numerical identifier which relates this node with its
		* 				position in a global matrix.
		* @param	x 	The node x-coordinate.
		* @param	y 	The node y-coordinate.
		* @param	z 	The node z-coordinate.
		*
		* ### param	numDofs	Number of dof's.
		**************************************************************************************************/
	Node(id_type id, coordtype x, coordtype y, coordtype z);

	/**********************************************************************************************//**
		* @fn	:::Node(id_type id, coordtype x, coordtype y);
		*
		* @brief	Creates a new node with the number of dof's given.
		*
		* @author	Renato T. Yamassaki
		* @date	05 abr 2012
		*
		* @param	id	Numerical identifier which relates this node with its
		* 				position in a global matrix.
		* @param	x 	The node x-coordinate.
		* @param	y 	The node y-coordinate.
		*
		* ### param	numDofs	Number of dof's.
		**************************************************************************************************/
	Node(id_type id, coordtype x, coordtype y);

	/**********************************************************************************************//**
		* @fn	:::Node(id_type internalId, id_type externalId, coordtype x,
		* 		coordtype y);
		*
		* @brief	Constructor.
		*
		* @author	Renato T. Yamassaki
		* @date	22 mar 2011
		*
		* @param	internalId	The identifier.
		* @param	externalId	External identifier for this node.
		* @param	x		  	The node x-coordinate.
		* @param	y		  	The node y-coordinate.
		*
		* ### param	numDofs	Number of dof's.
		**************************************************************************************************/
	Node(id_type internalId, id_type externalId, coordtype x, coordtype y);

	/**********************************************************************************************//**
		* @fn	:::Node(id_type internalId, id_type externalId, coordtype x,
		* 		coordtype y, coordtype z);
		*
		* @brief	Constructor.
		*
		* @author	Renato T. Yamassaki
		* @date	22 mar 2011
		*
		* @param	internalId	The identifier.
		* @param	externalId	External identifier for this node.
		* @param	x		  	The node x-coordinate.
		* @param	y		  	The node y-coordinate.
		* @param	z		  	The node z-coordinate.
		*
		* ### param	numDofs	Number of dof's.
		**************************************************************************************************/
	Node(id_type internalId, id_type externalId, coordtype x, coordtype y, coordtype z);

	/**********************************************************************************************//**
		* @brief	Destroys this object.
		*
		* @author	Renato T. Yamassaki
		* @date	20 ago 2012
		**************************************************************************************************/
	~Node(void);

	/**********************************************************************************************//**
		* @brief	Destroys this object.
		*
		* @author	Renato T. Yamassaki
		* @date	20 ago 2012
		**************************************************************************************************/
	void Destroy(void) const;


	/**********************************************************************************************//**
		* @fn	const coordtype& :::X(void) const;
		*
		* @brief	Gets the x coordinate.
		*
		* @author	Renato T. Yamassaki
		* @date	05 jun 2012
		*
		* @return	The x-coordinate of this object.
		**************************************************************************************************/
	const coordtype& X(void) const;

	/**********************************************************************************************//**
		* @fn	const coordtype& :::Y(void) const;
		*
		* @brief	Gets the y coordinate.
		*
		* @author	Renato T. Yamassaki
		* @date	05 jun 2012
		*
		* @return	The y-coordinate of this object.
		**************************************************************************************************/
	const coordtype& Y(void) const;

	/**********************************************************************************************//**
		* @fn	const coordtype& :::Z(void) const;
		*
		* @brief	Gets the z coordinate.
		*
		* @author	Renato T. Yamassaki
		* @date	05 jun 2012
		*
		* @return	The z-coordinate of this object.
		**************************************************************************************************/
	const coordtype& Z(void) const;

	/**********************************************************************************************//**
		* @fn	coordtype& :::X(void);
		*
		* @brief	Gets the x coordinate.
		*
		* @author	Renato T. Yamassaki
		* @date	05 jun 2012
		*
		* @return	The x-coordinate of this object.
		**************************************************************************************************/
	coordtype& X(void);

	/**********************************************************************************************//**
		* @fn	coordtype& :::Y(void);
		*
		* @brief	Gets the y coordinate.
		*
		* @author	Renato T. Yamassaki
		* @date	05 jun 2012
		*
		* @return	The y-coordinate of this object.
		**************************************************************************************************/
	coordtype& Y(void);

	/**********************************************************************************************//**
		* @fn	coordtype& :::Z(void);
		*
		* @brief	Gets the z coordinate.
		*
		* @author	Renato T. Yamassaki
		* @date	05 jun 2012
		*
		* @return	The z-coordinate of this object.
		**************************************************************************************************/
	coordtype& Z(void);

  coordtype& CurrentX(void);
  coordtype CurrentX(void) const;

  coordtype& CurrentY(void);
  coordtype CurrentY(void) const;

  coordtype& CurrentZ(void);
  coordtype CurrentZ(void) const;


	void InitDofs(int dofCount, DoF::id_type startId);

	/// <summary>
	/// Returns an <see cref="DoF" /> object representing a degree-of-freedom of this node given by its index.
	/// </summary>
	/// <param name="name">The zero-based index of the degree-of-freedom.</param>
	/// <exception cref="OutOfBoundsException">
	/// When the index is greater than or equal the total number of dof's.
	/// </exception>
	const DoF& GetDoF(int index) const;
  DoF& GetDoF(int index);

	/// <summary>
	/// Returns the total number of degrees-of-freedom of this node.
	/// </summary>
	int GetDofCount(void) const;

	bool WasInitialized(void) const;

	/// <summary>
	/// Returns a degree of freedom of this node.
	/// </summary>
	/// <param name="index">Zero-based index of the dof related to this node.</param>
	const DoF& operator [](int index) const;
  DoF& operator [](int index);

	id_type GetUserId(void) const;

	id_type GetInternalId(void) const;

	void ConnectElement( axis::foundation::memory::RelativePointer& element );

	int GetConnectedElementCount( void ) const;
  void CompileConnectivityList(void);

	axis::domain::elements::FiniteElement& GetConnectedElement( int elementIndex ) const;

  axis::foundation::blas::ColumnVector& Strain(void);
  const axis::foundation::blas::ColumnVector& Strain(void) const;

	axis::foundation::blas::ColumnVector& Stress(void);
	const axis::foundation::blas::ColumnVector& Stress(void) const;

	void ResetStrain(void);
	void ResetStress(void);

	static axis::foundation::memory::RelativePointer Create(id_type id);
	static axis::foundation::memory::RelativePointer Create(id_type internalId, id_type externalId);
	static axis::foundation::memory::RelativePointer Create(id_type id, coordtype x, coordtype y, coordtype z);
	static axis::foundation::memory::RelativePointer Create(id_type id, coordtype x, coordtype y);
	static axis::foundation::memory::RelativePointer Create(id_type internalId, id_type externalId, coordtype x, coordtype y);
	static axis::foundation::memory::RelativePointer Create(id_type internalId, id_type externalId, coordtype x, coordtype y, coordtype z);
  void *operator new (size_t bytes);
  void *operator new (size_t bytes, void *ptr);
  void operator delete(void *ptr);
  void operator delete(void *, void *);
private:
	axis::foundation::memory::RelativePointer _dofs[6];	// degrees-of-freedom for this node
  axis::foundation::memory::RelativePointer _strain;
  axis::foundation::memory::RelativePointer _stress;
  axis::foundation::memory::RelativePointer reverseConnList_;
	id_type _internalId;
	id_type _externalId;
	axis::domain::collections::ElementSet *_elements;
	coordtype _x, _y, _z;
  coordtype curX_, curY_, curZ_;
	int _numDofs;		// number of dof's
  bool isConnectivityListLocked_;

	/**********************************************************************************************//**
		* @fn	void :::initMembers(id_type internalId, id_type externalId,
		* 		coordtype x, coordtype y, coordtype z);
		*
		* @brief	Initializes internal members of this object.
		*
		* @author	Renato T. Yamassaki
		* @date	05 jun 2012
		*
		* @param	internalId	The number of degrees of freedom in the node.
		* @param	externalId	Identifier for the external.
		* @param	x		  	The node x-coordinate.
		* @param	y		  	The node y-coordinate.
		* @param	z		  	The node z-coordinate.
		**************************************************************************************************/
	void initMembers(id_type internalId, id_type externalId, coordtype x, coordtype y, coordtype z);

  Node(const Node& node);
  Node& operator=(const Node& n);
};

} } }
