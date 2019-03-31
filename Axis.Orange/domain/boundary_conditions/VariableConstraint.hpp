#pragma once
#include "domain/curves/Curve.hpp"
#include "domain/boundary_conditions/BoundaryCondition.hpp"

namespace axis { namespace domain { namespace boundary_conditions {

/**************************************************************************************************
	* <summary>	Describes any boundary condition that can vary along time. </summary>
	**************************************************************************************************/
class VariableConstraint : public BoundaryCondition
{
public:
	/**********************************************************************************************//**
		* <summary> Creates a new instance of this class.</summary>
		*
		* <param name="type">    The constraint type.</param>
		* <param name="history"> Load behavior along time.</param>
		**************************************************************************************************/
	VariableConstraint(ConstraintType type, axis::foundation::memory::RelativePointer& history);

	/**********************************************************************************************//**
		* <summary> Creates a new instance of this class.</summary>
		*
		* <param name="type">    The constraint type.</param>
		* <param name="history"> Load behavior along time.</param>
		* <param name="scalar">  Load value.</param>
		**************************************************************************************************/
	VariableConstraint(ConstraintType type, axis::foundation::memory::RelativePointer& history, 
    real scalar);

	/**********************************************************************************************//**
		* <summary> Creates a new instance of this class.</summary>
		*
		* <param name="type">    The constraint type.</param>
		* <param name="history"> Load behavior along time.</param>
		* <param name="scalar">  Load value.</param>
    * <param name="releaseTime">  Time when boundary condition is no longer active.</param>
		**************************************************************************************************/
	VariableConstraint(ConstraintType type, axis::foundation::memory::RelativePointer& history, 
    real scalar, real releaseTime);

	/// <summary>
	/// Destroys this object.
	/// </summary>
	~VariableConstraint(void);
	virtual BoundaryCondition& Clone(void) const;
	virtual void Destroy( void ) const;
	virtual real GetValue( real time ) const;
	virtual bool IsNonNodalLoad(void) const;
	virtual bool IsVariableConstraint(void) const;
	virtual bool IsNodalLock(void) const;
  virtual bool Active(real time) const;
  virtual bool IsGPUCapable( void ) const;
  virtual axis::foundation::uuids::Uuid GetTypeId( void ) const;

  virtual BoundaryConditionUpdateCommand& GetUpdateCommand( void );

  virtual int GetGPUDataSize( void ) const;

  virtual void InitGPUData( void *data, real& outputBucket );
private:
	axis::foundation::memory::RelativePointer history_;
	real scalar_;
  real releaseTime_;
};

} } } // namespace axis::domain::boundary_conditions
