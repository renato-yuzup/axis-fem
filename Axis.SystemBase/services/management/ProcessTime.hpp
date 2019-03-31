#pragma once
#include "foundation/Axis.SystemBase.hpp"
#include "foundation/date_time/Timestamp.hpp"
#include "foundation/date_time/Timespan.hpp"

namespace axis
{
	namespace services
	{
		namespace management
		{
			/**********************************************************************************************//**
			 * @brief	Stores information about processing time.
			 *
			 * @author	Renato T. Yamassaki
			 * @date	27 ago 2012
			 **************************************************************************************************/
			class AXISSYSTEMBASE_API ProcessTime
			{
			private:
				axis::foundation::date_time::Timestamp _creationTime;
				axis::foundation::date_time::Timespan _userTime;
				axis::foundation::date_time::Timespan _kernelTime;
			public:

				/**********************************************************************************************//**
				 * @brief	Default constructor. Initializes all timings to zero.
				 *
				 * @author	Renato T. Yamassaki
				 * @date	27 ago 2012
				 **************************************************************************************************/
				ProcessTime(void);

				/**********************************************************************************************//**
				 * @brief	Creates a new object.
				 *
				 * @author	Renato T. Yamassaki
				 * @date	27 ago 2012
				 *
				 * @param	creationTime	Current process creation time.
				 * @param	userTime		Total time that the current process 
				 * 							stayed on user mode.
				 * @param	kernelTime  	Total time that the current process
				 * 							stayed on kernel mode.
				 **************************************************************************************************/
				ProcessTime(axis::foundation::date_time::Timestamp creationTime, 
							axis::foundation::date_time::Timespan userTime, 
							axis::foundation::date_time::Timespan kernelTime);

				/**********************************************************************************************//**
				 * @brief	Destructor.
				 *
				 * @author	Renato T. Yamassaki
				 * @date	27 ago 2012
				 **************************************************************************************************/
				~ProcessTime(void);

				/**********************************************************************************************//**
				 * @brief	Returns the creation time of the current process.
				 *
				 * @author	Renato T. Yamassaki
				 * @date	27 ago 2012
				 *
				 * @return	An object containing the creation time.
				 **************************************************************************************************/
				axis::foundation::date_time::Timestamp CreationTime(void) const;

				/**********************************************************************************************//**
				 * @brief	Returns the total user time of all threads in the 
				 * 			current process.
				 *
				 * @author	Renato T. Yamassaki
				 * @date	27 ago 2012
				 *
				 * @return	An object containing the user time.
				 **************************************************************************************************/
				axis::foundation::date_time::Timespan UserTime(void) const;

				/**********************************************************************************************//**
				 * @brief	Returns the total kernel time of all threads in the 
				 * 			current process.
				 *
				 * @author	Renato T. Yamassaki
				 * @date	27 ago 2012
				 *
				 * @return	An object containing the kernel time.
				 **************************************************************************************************/
				axis::foundation::date_time::Timespan KernelTime(void) const;
			};		
		}
	}
}

