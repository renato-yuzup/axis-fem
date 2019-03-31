#pragma once
#include "foundation/Axis.CommonLibrary.hpp"
#include "foundation/collections/Collectible.hpp"
#include "domain/elements/DoF.hpp"
#include "foundation/Axis.SystemBase.hpp"
#include "foundation/memory/RelativePointer.hpp"
#include "foundation/uuids/Uuid.hpp"
#include "BoundaryConditionUpdateCommand.hpp"

namespace axis
{
	namespace domain
	{
		namespace boundary_conditions
		{
			class AXISCOMMONLIBRARY_API BoundaryCondition : public axis::foundation::collections::Collectible
			{
			public:
				enum ConstraintType
				{
					PrescribedDisplacement = 0,
					PrescribedVelocity = 1,
					NodalLoad = 2,
					Lock = 3,
          PrescribedAcceleration = 4
				};
			private:
				ConstraintType _type;
			protected:
				axis::domain::elements::DoF *_dof;
			public:
				/**********************************************************************************************//**
				 * <summary> Constructor.</summary>
				 *
				 * <param name="type"> The constraint type.</param>
				 **************************************************************************************************/
				BoundaryCondition(ConstraintType type);

				virtual ~BoundaryCondition(void);

				/**********************************************************************************************//**
				 * @fn	axis::domain::elements::DoF :::*GetDoF(void) const;
				 *
				 * @brief	Returns the degree of freedom to which this boundary
				 * 			condition is applied.
				 *
				 * @author	Renato T. Yamassaki
				 * @date	05 jun 2012
				 *
				 * @return	null if it fails, else the dof.
				 **************************************************************************************************/
				axis::domain::elements::DoF *GetDoF(void) const;

				/**********************************************************************************************//**
				 * @fn	void :::SetDoF(axis::domain::elements::DoF *dof);
				 *
				 * @brief	Sets the degree of freedom to which this boundary
				 * 			condition is applied.
				 *
				 * @author	Renato T. Yamassaki
				 * @date	05 jun 2012
				 *
				 * @param [in,out]	dof	The dof to which this boundary condition is
				 * 						applied.
				 **************************************************************************************************/
				void SetDoF(axis::domain::elements::DoF *dof);

				bool IsLoad(void) const;
				bool IsLock(void) const;
				bool IsPrescribedDisplacement(void) const;
				bool IsPrescribedVelocity(void) const;


				/**************************************************************************************************
				 * <summary>	Tells if the boundary condition is active in specified time.</summary>
				 *
				 * <returns>	true if it is active, false otherwise. </returns>
				 **************************************************************************************************/        
        virtual bool Active(real time) const = 0;

				/**************************************************************************************************
				 * <summary>	Makes a deep copy of this object. </summary>
				 *
				 * <returns>	A copy of this object. </returns>
				 **************************************************************************************************/
				virtual BoundaryCondition& Clone(void) const = 0;

				/**********************************************************************************************//**
				 * @fn	virtual void :::Destroy(void) const = 0;
				 *
				 * @brief	Destroys this object.
				 *
				 * @author	Renato
				 * @date	08/06/2012
				 **************************************************************************************************/
				virtual void Destroy(void) const = 0;

				/**********************************************************************************************//**
				 * <summary> Returns the value of the boundary condition in a given instant of time.</summary>
				 *
				 * <param name="time"> The instant of time.</param>
				 *
				 * <returns> The boundary condition value.</returns>
				 **************************************************************************************************/
				virtual real GetValue(real time) const = 0;

        real operator()(real time) const;

				/**********************************************************************************************//**
				 * <summary> Returns the type of this boundary condition.</summary>
				 *
				 * <returns> The boundary condition type of this object.</returns>
				 **************************************************************************************************/
				ConstraintType GetType(void) const;

        /**
         * Returns if this boundary condition object can execute on GPU devices.
         *
         * @return true if it is GPU capable, false otherwise.
        **/
        virtual bool IsGPUCapable(void) const;

        virtual axis::foundation::uuids::Uuid GetTypeId(void) const = 0;

        virtual BoundaryConditionUpdateCommand& GetUpdateCommand(void);

        virtual int GetGPUDataSize(void) const;
        virtual void InitGPUData(void *data, real& outputBucket);
			};
		}
	}
}
