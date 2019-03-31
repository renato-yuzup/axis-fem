#pragma once
#include "foundation/Axis.SystemBase.hpp"
#include "foundation/date_time/Timespan.hpp"
#include "foundation/date_time/Timestamp.hpp"

namespace axis
{
	namespace services
	{
		namespace diagnostics
		{
			/**************************************************************************************************
			 * <summary>	Provides information about a process in the operating system. </summary>
			 **************************************************************************************************/
			class AXISSYSTEMBASE_API Process
			{
			private:
				class ProcessData;
				ProcessData *_data;

				/**************************************************************************************************
				 * <summary>	Private constructor. For internal use only </summary>
				 *
				 * <param name="processId">	Identifier for the process this object will monitor. </param>
				 **************************************************************************************************/
				Process(long processId);
			public:

				/**
				 * Creates a new instance of this class pointing to an invalid process.
				 */
				Process(void);

				/**************************************************************************************************
				 * <summary>	Copy constructor. </summary>
				 *
				 * <param name="other">	The other process object. </param>
				 **************************************************************************************************/
				Process(const Process& other);

				/**************************************************************************************************
				 * <summary>	Destructor. </summary>
				 **************************************************************************************************/
				~Process(void);

				/**************************************************************************************************
				 * <summary>	Copy assignment operator. </summary>
				 *
				 * <param name="other">	The other process object. </param>
				 *
				 * <returns>	A reference to this object. </returns>
				 **************************************************************************************************/
				Process& operator = (const Process& other);

				/**************************************************************************************************
				 * <summary>	Returns an object that monitors the current process. </summary>
				 *
				 * <returns>	The process object. </returns>
				 **************************************************************************************************/
				static Process GetCurrentProcess(void);

				/**************************************************************************************************
				 * <summary>	Updates information about the process. </summary>
				 **************************************************************************************************/
				void Refresh(void);

				/**************************************************************************************************
				 * <summary>	Returns the process identifier. </summary>
				 *
				 * <returns>	The process identifier. </returns>
				 **************************************************************************************************/
				long GetProcessId(void) const;

				/**************************************************************************************************
				 * <summary>	Returns the amount of CPU time this process has spend in kernel calls among all
				 * 				threads in all CPUs. </summary>
				 *
				 * <returns>	The kernel time. </returns>
				 **************************************************************************************************/
				axis::foundation::date_time::Timespan GetKernelTime(void) const;

				/**************************************************************************************************
				 * <summary>	Returns the amount of CPU time this process has spend in user code among all
				 * 				threads in all CPUs. </summary>
				 *
				 * <returns>	The user time. </returns>
				 **************************************************************************************************/
				axis::foundation::date_time::Timespan GetUserTime(void) const;

				/**************************************************************************************************
				 * <summary>	Returns how much time has elapsed since the process was created. </summary>
				 *
				 * <returns>	The wall time. </returns>
				 **************************************************************************************************/
				axis::foundation::date_time::Timespan GetWallTime(void) const;

				/**************************************************************************************************
				 * <summary>	Returns the process creation time. </summary>
				 *
				 * <returns>	The creation time. </returns>
				 **************************************************************************************************/
				axis::foundation::date_time::Timestamp GetCreationTime(void) const;

				/**************************************************************************************************
				 * <summary>	Returns how much physical memory is allocated for this process. </summary>
				 *
				 * <returns>	The physical memory allocated size in bytes. </returns>
				 **************************************************************************************************/
				uint64 GetPhysicalMemoryAllocated(void) const;

				/**************************************************************************************************
				 * <summary>	Returns how much virtual memory is allocated for this process. </summary>
				 *
				 * <returns>	The virtual memory allocated size in bytes. </returns>
				 **************************************************************************************************/
				uint64 GetVirtualMemoryAllocated(void) const;

				/**************************************************************************************************
				 * <summary>	Returns how much paged memory is allocated for this process. </summary>
				 *
				 * <returns>	The paged memory allocated size in bytes. </returns>
				 **************************************************************************************************/
				uint64 GetPagedMemoryAllocated(void) const;

				/**************************************************************************************************
				 * <summary>	Returns the maximum of physical memory allocated for this process. </summary>
				 *
				 * <returns>	The peak physical memory allocated size, in bytes. </returns>
				 **************************************************************************************************/
				uint64 GetPeakPhysicalMemoryAllocated(void) const;

				/**************************************************************************************************
				 * <summary>	Returns the maximum of virtual memory allocated for this process. </summary>
				 *
				 * <returns>	The peak virtual memory allocated size, in bytes. </returns>
				 **************************************************************************************************/
				uint64 GetPeakVirtualMemoryAllocated(void) const;

				/**************************************************************************************************
				 * <summary>	Returns the maximum of paged memory allocated for this process. </summary>
				 *
				 * <returns>	The peak paged memory allocated size, in bytes. </returns>
				 **************************************************************************************************/
				uint64 GetPeakPagedMemoryAllocated(void) const;

				/**************************************************************************************************
				 * <summary>	Returns the percentage of CPU time that this process has consumed compared to
				 * 				other processes in the system, since last refresh. </summary>
				 *
				 * <returns>	The CPU usage percent (maximum value is 1). </returns>
				 **************************************************************************************************/
				double GetCPUUsage(void) const;

				/**************************************************************************************************
				 * <summary>	Returns a percentage that represents the CPU occupation for this process, which is
				 * 				the CPU time of this process compared to other processes in the system, including
				 * 				idle time, since last refresh. </summary>
				 *
				 * <returns>	The CPU occupation percent (maximum value is 1). </returns>
				 **************************************************************************************************/
				double GetCPUOccupation(void) const;

				/**************************************************************************************************
				 * <summary>	Returns a percentage of how much this process has spent its CPU time in user 
				 * 				code execution, since last refresh. </summary>
				 *
				 * <returns>	The user code occupation percent (maximum value is 1). </returns>
				 **************************************************************************************************/
				double GetUserCodeOccupation(void) const;

				/**************************************************************************************************
				 * <summary>	Returns the percentage of CPU time that this process has consumed compared to
				 * 				other processes in the system, since its creation. </summary>
				 *
				 * <returns>	The total CPU usage percent (maximum value is 1). </returns>
				 **************************************************************************************************/
				double GetTotalCPUUsage(void) const;

				/**************************************************************************************************
				 * <summary>	Returns a percentage that represents the CPU occupation for this process, which is
				 * 				the CPU time of this process compared to other processes in the system, including
				 * 				idle time, since its creation. </summary>
				 *
				 * <returns>	The total CPU occupation percent (maximum value is 1). </returns>
				 **************************************************************************************************/
				double GetTotalCPUOccupation(void) const;

				/**************************************************************************************************
				 * <summary>	Returns a percentage of how much this process has spent its CPU time in user 
				 * 				code execution, since its creation. </summary>
				 *
				 * <returns>	The total user code occupation percent (maximum value is 1). </returns>
				 **************************************************************************************************/
				double GetTotalUserCodeOccupation(void) const;
			};
		}
	}
}

