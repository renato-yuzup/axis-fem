#pragma once
#include "yuzu/common/gpu.hpp"
#include "yuzu/domain/elements/DoF.hpp"
#include "yuzu/foundation/blas/ColumnVector.hpp"

namespace axis { namespace yuzu { namespace domain { namespace elements {

class FiniteElement;

/// <summary>
/// Represents a node used to connect and define finite elements.
/// </summary>
class Node
{
public:
	typedef long id_type;

  /**********************************************************************************************//**
		* @brief	Destroys this object.
		*
		* @author	Renato T. Yamassaki
		* @date	20 ago 2012
		**************************************************************************************************/
	GPU_ONLY ~Node(void);

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
	GPU_ONLY const coordtype& X(void) const;

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
	GPU_ONLY const coordtype& Y(void) const;

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
	GPU_ONLY const coordtype& Z(void) const;

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
	GPU_ONLY coordtype& X(void);

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
	GPU_ONLY coordtype& Y(void);

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
	GPU_ONLY coordtype& Z(void);

  GPU_ONLY coordtype& CurrentX(void);
  GPU_ONLY coordtype CurrentX(void) const;

  GPU_ONLY coordtype& CurrentY(void);
  GPU_ONLY coordtype CurrentY(void) const;

  GPU_ONLY coordtype& CurrentZ(void);
  GPU_ONLY coordtype CurrentZ(void) const;

  /// <summary>
	/// Returns an <see cref="DoF" /> object representing a degree-of-freedom of this node given by its index.
	/// </summary>
	/// <param name="name">The zero-based index of the degree-of-freedom.</param>
	/// <exception cref="OutOfBoundsException">
	/// When the index is greater than or equal the total number of dof's.
	/// </exception>
	GPU_ONLY const DoF& GetDoF(int index) const;
  GPU_ONLY DoF& GetDoF(int index);

	/// <summary>
	/// Returns the total number of degrees-of-freedom of this node.
	/// </summary>
	GPU_ONLY int GetDofCount(void) const;

	GPU_ONLY bool WasInitialized(void) const;

	/// <summary>
	/// Returns a degree of freedom of this node.
	/// </summary>
	/// <param name="index">Zero-based index of the dof related to this node.</param>
	GPU_ONLY const DoF& operator [](int index) const;
  GPU_ONLY DoF& operator [](int index);

	GPU_ONLY id_type GetUserId(void) const;

	GPU_ONLY id_type GetInternalId(void) const;

	GPU_ONLY int GetConnectedElementCount( void ) const;

	GPU_ONLY axis::yuzu::domain::elements::FiniteElement& GetConnectedElement( int elementIndex ) const;

  GPU_ONLY axis::yuzu::foundation::blas::ColumnVector& Strain(void);
  GPU_ONLY const axis::yuzu::foundation::blas::ColumnVector& Strain(void) const;

	GPU_ONLY axis::yuzu::foundation::blas::ColumnVector& Stress(void);
	GPU_ONLY const axis::yuzu::foundation::blas::ColumnVector& Stress(void) const;

	GPU_ONLY void ResetStrain(void);
	GPU_ONLY void ResetStress(void);
private:
  axis::yuzu::foundation::memory::RelativePointer _dofs[6];	// degrees-of-freedom for this node
  axis::yuzu::foundation::memory::RelativePointer _strain;
  axis::yuzu::foundation::memory::RelativePointer _stress;
  axis::yuzu::foundation::memory::RelativePointer reverseConnList_;
  id_type _internalId;
  id_type _externalId;
  volatile void *_elements;
  coordtype _x; coordtype _y; coordtype _z;
  coordtype curX_, curY_, curZ_;
  int _numDofs;		// number of dof's
  bool isConnectivityListLocked_;

  Node(void);
  Node(const Node& node);
  Node& operator=(const Node& n);
};

} } } }