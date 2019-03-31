#pragma once
#include "foundation/Axis.CommonLibrary.hpp"
#include "foundation/collections/ObjectList.hpp"
#include "SnapshotMark.hpp"

namespace axis { namespace domain {	namespace analyses {

class AXISCOMMONLIBRARY_API AnalysisTimeline
{
public:
	/**********************************************************************************************//**
		* <summary> Defines an alias for the iteration index numeric type.</summary>
		**************************************************************************************************/
	typedef unsigned long iteration_index;

	/**********************************************************************************************//**
		* <summary> Creates a new instance of this class.</summary>
		*
		* <param name="startTime"> The start time of this timeline.</param>
		* <param name="endTime">   The end time of this timeline.</param>
		*
		* <returns> A new instance of this class.</returns>
		**************************************************************************************************/
	static AnalysisTimeline& Create(real startTime, real endTime);

	/**********************************************************************************************//**
		* <summary> Creates a new instance of this class.</summary>
		*
		* <param name="startTime">		    The start time of this timeline.</param>
		* <param name="endTime">		    The end time of this timeline.</param>
		* <param name="currentTime">	    The current time.</param>
		* <param name="lastTimeIncrement"> The last time increment.</param>
		*
		* <returns> A new instance of this class.</returns>
		**************************************************************************************************/
	static AnalysisTimeline& Create(real startTime, real endTime, real currentTime, real lastTimeIncrement);

	/**********************************************************************************************//**
		* <summary> Copy constructor.</summary>
		*
		* <param name="other"> The other object.</param>
		**************************************************************************************************/
	AnalysisTimeline(const AnalysisTimeline& other);

	/**********************************************************************************************//**
		* <summary> Destructor.</summary>
		**************************************************************************************************/
	~AnalysisTimeline(void);

	/**********************************************************************************************//**
		* <summary> Destroys this object.</summary>
		**************************************************************************************************/
	void Destroy(void) const;

	/**********************************************************************************************//**
		* <summary> Returns the time iteration index.</summary>
		*
		* <returns> The zero-based index of the current iteration.</returns>
		**************************************************************************************************/
	iteration_index IterationIndex(void) const;

	/**********************************************************************************************//**
		* <summary> Returns the time iteration index.</summary>
		*
		* <returns> A writable reference to the time iteration index, a zero-based
		* 			 index of the current iteration.</returns>
		**************************************************************************************************/
	iteration_index& IterationIndex(void);

	/**********************************************************************************************//**
		* <summary> Returns the hypothetical start time of a numerical analysis.</summary>
		*
		* <returns> A real number.</returns>
		**************************************************************************************************/
	real StartTime(void) const;


	/**********************************************************************************************//**
		* <summary> Returns the hypothetical end time of a numerical analysis.</summary>
		*
		* <returns> A real number.</returns>
		**************************************************************************************************/
	real EndTime(void) const;

	/**********************************************************************************************//**
		* <summary> Returns by how much the current analysis time will be incremented in the next 
		* 			 iteration.</summary>
		*
		* <returns> A real number.</returns>
		**************************************************************************************************/
	real NextTimeIncrement(void) const;

	/**********************************************************************************************//**
		* <summary> Returns by how much the current analysis time will be incremented in the next 
		* 			 time iteration.</summary>
		*
		* <returns> A writable reference to a real number.</returns>
		**************************************************************************************************/
	real& NextTimeIncrement(void);

	/**********************************************************************************************//**
		* <summary> Returns by how much the current analysis time was incremented from the last
		* 			 time iteration.</summary>
		*
		* <returns> A real number.</returns>
		**************************************************************************************************/
	real LastTimeIncrement(void) const;

	/**********************************************************************************************//**
		* <summary> Returns the current iteration position in this timeline.</summary>
		*
		* <returns> A real number.</returns>
		**************************************************************************************************/
	real GetCurrentTimeMark(void) const;

	/**********************************************************************************************//**
		* <summary> Increments iteration index and current time.</summary>
		* <remarks> Iteration index is incremented by 1 and current time is incremented by the value
		* 			 specified in NextTimeIncrement. After this, LastTimeIncrement holds the value of
		* 			 NextTimeIncrement, which will be equal to zero after this call.
		**************************************************************************************************/
	void Tick(void);

	/**********************************************************************************************//**
		* <summary> Adds a time mark which defines the main steps of the analysis.</summary>
		*
		* <param name="mark"> The mark.</param>
		* <remarks> Marks should be added in ascending order, or else an error occurs.</remarks>
		**************************************************************************************************/
	void AddSnapshotMark(const SnapshotMark& mark);

	/**********************************************************************************************//**
		* <summary> Clears the snapshot marks.</summary>
		**************************************************************************************************/
	void ClearSnapshotMarks(void);

	/**********************************************************************************************//**
		* <summary> Returns the snapshot mark count.</summary>
		*
		* <returns> A non-negative integer.</returns>
		**************************************************************************************************/
	size_type SnapshotMarkCount(void) const;

	/**********************************************************************************************//**
		* <summary> Returns the time position of the current snapshot mark.</summary>
		*
		* <returns> The current snapshot time.</returns>
		* <remarks> If no snapshot mark has been crossed in the last tick operation,
		* 			 an error occurs.</remarks>
		**************************************************************************************************/
	real GetCurrentSnapshotTime(void) const;

	/**********************************************************************************************//**
		* <summary> Queries if a snapshot mark has been crossed in the last
		* 			 tick operation.
		*
		* <returns> true if it has crossed, false otherwise.</returns>
		**************************************************************************************************/
	bool HasCrossedSnapshotMark(void) const;

	/**********************************************************************************************//**
		* <summary> Resets time position in this timelime.</summary>
		**************************************************************************************************/
	void Reset(void);

	/**************************************************************************************************
		* <summary>	Returns a snapshot mark inserted in the timeline. </summary>
		*
		* <param name="index">	Zero-based index of the mark. </param>
		*
		* <returns>	The snapshot mark. </returns>
		**************************************************************************************************/
	SnapshotMark& GetSnapshotMark(size_type index) const;

	/**********************************************************************************************//**
		* <summary> Copy assignment operator.</summary>
		*
		* <param name="other"> The other object.</param>
		*
		* <returns> A reference to this object.</returns>
		**************************************************************************************************/
	AnalysisTimeline& operator =(const AnalysisTimeline& other);
private:
  class Pimpl;
  AnalysisTimeline(real startTime, real endTime);
  AnalysisTimeline(real startTime, real endTime, real currentTime, real lastTimeIncrement);
  void Init(void);
  void Copy(const AnalysisTimeline& other);

  Pimpl *pimpl_;
  // time definitions
  iteration_index iterationIndex_;
  real lastTimestep_;
  real nextTimestep_;
  real startTime_;
  real endTime_;
  real currentTime_;
};

}	} }
