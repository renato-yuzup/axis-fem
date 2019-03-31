#pragma once
#include "foundation/Axis.CommonLibrary.hpp"
#include "foundation/collections/Collectible.hpp"

namespace axis
{
	namespace domain
	{
		namespace analyses
		{
			/**********************************************************************************************//**
			 * <summary> Defines a time mark along the analysis timeline where
			 * 			 data collection should occur.</summary>
			 **************************************************************************************************/
			class AXISCOMMONLIBRARY_API SnapshotMark : public axis::foundation::collections::Collectible
			{
			private:
				real _time;
			public:

				/**********************************************************************************************//**
				 * <summary> Constructor.</summary>
				 *
				 * <param name="time"> The time for this mark.</param>
				 **************************************************************************************************/
				SnapshotMark(real time);

				/**********************************************************************************************//**
				 * <summary> Destructor.</summary>
				 **************************************************************************************************/
				~SnapshotMark(void);

				/**********************************************************************************************//**
				 * <summary> Destroys this object.</summary>
				 **************************************************************************************************/
				void Destroy(void) const;

				/**********************************************************************************************//**
				 * <summary> Gets the time of this mark.</summary>
				 *
				 * <returns> A real number.</returns>
				 **************************************************************************************************/
				real GetTime(void) const;

				/**********************************************************************************************//**
				 * <summary> Makes a deep copy of this object.</summary>
				 *
				 * <returns> A copy of this object.</returns>
				 **************************************************************************************************/
				SnapshotMark& Clone(void) const;
			};		
		}
	}
}

