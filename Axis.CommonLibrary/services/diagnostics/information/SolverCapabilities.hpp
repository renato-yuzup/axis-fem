#pragma once
#include "foundation/Axis.CommonLibrary.hpp"
#include "AxisString.hpp"

namespace axis
{
	namespace services
	{
		namespace diagnostics
		{
			namespace information
			{
				/**********************************************************************************************//**
				 * <summary> Describes the capabilities of a solver.</summary>
				 **************************************************************************************************/
				class AXISCOMMONLIBRARY_API SolverCapabilities
				{
				private:
					bool _timeCap, _materialCap, _geometricCap, _bcCap;
					axis::String _name;
					axis::String _description;

					void Copy(const SolverCapabilities& other);
				public:

					/**********************************************************************************************//**
					 * <summary> Constructor.</summary>
					 *
					 * <param name="solverName">	  Name of the solver.</param>
					 * <param name="description">	  The solver description.</param>
					 * <param name="timeDepedentCap"> true to capable of solving time depedent problems.</param>
					 * <param name="materialCap">	  true to capable of accounting material nonlinearity.</param>
					 * <param name="geometricCap">    true to capable of accounting geometric nonlinearity.</param>
					 * <param name="bcCap">			  true to capable of accounting boundary conditions nonlinearity.</param>
					 **************************************************************************************************/
					SolverCapabilities(const axis::String& solverName, 
									   const axis::String& description, 
									   bool timeDepedentCap, bool materialCap, 
									   bool geometricCap, bool bcCap);

					/**********************************************************************************************//**
					 * <summary> Copy constructor.</summary>
					 *
					 * <param name="other"> The other object.</param>
					 **************************************************************************************************/
					SolverCapabilities(const SolverCapabilities& other);

					/**********************************************************************************************//**
					 * <summary> Destructor.</summary>
					 **************************************************************************************************/
					~SolverCapabilities(void);

					/**********************************************************************************************//**
					 * <summary> Returns the solver name.</summary>
					 *
					 * <returns> The solver name.</returns>
					 **************************************************************************************************/
					axis::String GetSolverName(void) const;

					/**********************************************************************************************//**
					 * <summary> Returns the solver description.</summary>
					 *
					 * <returns> The description.</returns>
					 **************************************************************************************************/
					axis::String GetDescription(void) const;

					/**********************************************************************************************//**
					 * <summary> Determines if the solver can solve time dependent
					 * 			 problems.</summary>
					 *
					 * <returns> true if it does, false otherwise.</returns>
					 **************************************************************************************************/
					bool DoesSolveTimeDependentProblems(void) const;

					/**********************************************************************************************//**
					 * <summary> Determines if the solver accounts for material
					 * 			 nonlinearity.</summary>
					 *
					 * <returns> true if it does, false otherwise.</returns>
					 **************************************************************************************************/
					bool DoesAccountMaterialNonlinearity(void) const;

					/**********************************************************************************************//**
					 * <summary> Determines if the solver accounts for geometric
					 * 			 nonlinearity.</summary>
					 *
					 * <returns> true if it does, false otherwise.</returns>
					 **************************************************************************************************/
					bool DoesAccountGeometricNonlinearity(void) const;

					/**********************************************************************************************//**
					 * <summary> Determines if the solver accounts for boundary conditions
					 * 			 nonlinearity.</summary>
					 *
					 * <returns> true if it does, false otherwise.</returns>
					 **************************************************************************************************/
					bool DoesAccountBoundaryConditionsNonlinearity(void) const;

					/**********************************************************************************************//**
					 * <summary> Copy assignment operator.</summary>
					 *
					 * <param name="other"> The other object.</param>
					 *
					 * <returns> A reference to this object.</returns>
					 **************************************************************************************************/
					SolverCapabilities& operator = (const SolverCapabilities& other);
				};
			}
		}
	}
}
