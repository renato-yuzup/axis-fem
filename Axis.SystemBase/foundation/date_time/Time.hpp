#pragma once
#include "foundation/Axis.SystemBase.hpp"
#include "foundation/date_time/Timespan.hpp"

namespace axis
{
	namespace foundation
	{
		namespace date_time
		{
			/**************************************************************************************************
			 * <summary>	Class for time arithmetics and representation. Objects from this class are
			 * 				immutable although it does allow copy assignment. </summary>
			 **************************************************************************************************/
			class AXISSYSTEMBASE_API Time
			{
			private:
				class TimeData;
				TimeData *_data;
			public:
				/**************************************************************************************************
				 * <summary>	Creates a new time object pointing to midnight. </summary>
				 **************************************************************************************************/
				Time(void);

				/**************************************************************************************************
				 * <summary>	Copy constructor. </summary>
				 *
				 * <param name="other">	The other time object. </param>
				 **************************************************************************************************/
				Time(const Time& other);

				/**************************************************************************************************
				 * <summary>	Creates a new time object pointing to a specific instant. </summary>
				 *
				 * <param name="hours">  	The hours. </param>
				 * <param name="minutes">	The minutes. </param>
				 * <param name="seconds">	The seconds. </param>
				 **************************************************************************************************/
				Time(int hours, int minutes, int seconds);

				/**************************************************************************************************
				 * <summary>	Creates a new time object pointing to a specific instant. </summary>
				 *
				 * <param name="hours">		  	The hours. </param>
				 * <param name="minutes">	  	The minutes. </param>
				 * <param name="seconds">	  	The seconds. </param>
				 * <param name="milliseconds">	The milliseconds. </param>
				 **************************************************************************************************/
				Time(int hours, int minutes, int seconds, int milliseconds);

				/**************************************************************************************************
				 * <summary>	Destructor. </summary>
				 **************************************************************************************************/
				~Time(void);

				/**************************************************************************************************
				 * <summary>	Returns the hours. </summary>
				 *
				 * <returns>	The hours. </returns>
				 **************************************************************************************************/
				int GetHours(void) const;

				/**************************************************************************************************
				 * <summary>	Returns the minutes of the hour. </summary>
				 *
				 * <returns>	The minutes. </returns>
				 **************************************************************************************************/
				int GetMinutes(void) const;

				/**************************************************************************************************
				 * <summary>	Returns the seconds of the minute. </summary>
				 *
				 * <returns>	The seconds. </returns>
				 **************************************************************************************************/
				int GetSeconds(void) const;

				/**************************************************************************************************
				 * <summary>	Returns the the milliseconds of the second. </summary>
				 *
				 * <returns>	The milliseconds. </returns>
				 **************************************************************************************************/
				int GetMilliseconds(void) const;

				/**************************************************************************************************
				 * <summary>	Returns how many minutes has passed since midnight. </summary>
				 *
				 * <returns>	The total minutes. </returns>
				 **************************************************************************************************/
				long GetTotalMinutes(void) const;

				/**************************************************************************************************
				 * <summary>	Returns how many seconds has passed since midnight. </summary>
				 *
				 * <returns>	The total seconds. </returns>
				 **************************************************************************************************/
				long GetTotalSeconds(void) const;

				/**************************************************************************************************
				 * <summary>	Returns how many milliseconds has passed since midnight. </summary>
				 *
				 * <returns>	The total milliseconds. </returns>
				 **************************************************************************************************/
				long GetTotalMilliseconds(void) const;

				/**************************************************************************************************
				 * <summary>	Less-than comparison operator. </summary>
				 *
				 * <param name="other">	The other time to compare. </param>
				 *
				 * <returns>	true if this time is earlier than the other one. </returns>
				 **************************************************************************************************/
				bool operator < (const Time& other) const;

				/**************************************************************************************************
				 * <summary>	Less-than-or-equal comparison operator. </summary>
				 *
				 * <param name="other">	The other time to compare. </param>
				 *
				 * <returns>	true if this time is earlier than or equal to the other one. </returns>
				 **************************************************************************************************/
				bool operator <= (const Time& other) const;

				/**************************************************************************************************
				 * <summary>	Greater-than comparison operator. </summary>
				 *
				 * <param name="other">	The other time to compare. </param>
				 *
				 * <returns>	true if this time is after the other one. </returns>
				 **************************************************************************************************/
				bool operator > (const Time& other) const;

				/**************************************************************************************************
				 * <summary>	Greater-than-or-equal comparison operator. </summary>
				 *
				 * <param name="other">	The other time to compare. </param>
				 *
				 * <returns>	true if this time is equal to or after the other one. </returns>
				 **************************************************************************************************/
				bool operator >= (const Time& other) const;

				/**************************************************************************************************
				 * <summary>	Equality operator. </summary>
				 *
				 * <param name="other">	The other time to compare. </param>
				 *
				 * <returns>	true if this time is equal to the other one. </returns>
				 **************************************************************************************************/
				bool operator == (const Time& other) const;

				/**************************************************************************************************
				 * <summary>	Inequality operator. </summary>
				 *
				 * <param name="other">	The other time to compare. </param>
				 *
				 * <returns>	true if this time is different from the other one. </returns>
				 **************************************************************************************************/
				bool operator != (const Time& other) const;

				/**************************************************************************************************
				 * <summary>	Copy assignment operator. </summary>
				 *
				 * <param name="other">	The other time object. </param>
				 *
				 * <returns>	A reference to this object. </returns>
				 **************************************************************************************************/
				Time& operator = (const Time& other);

				/**************************************************************************************************
				 * <summary>	Returns the current time. </summary>
				 *
				 * <returns>	A new time object pointing to current time. </returns>
				 **************************************************************************************************/
				static Time Now(void);
			};

			/**************************************************************************************************
			 * <summary>	Addition operator. </summary>
			 *
			 * <param name="time">	  	The base time. </param>
			 * <param name="timespan">	Amount of time to add. </param>
			 *
			 * <returns>
			 *  A new time object farther from the base time by the amount specified in the timespan
			 *  parameter. Whole days are not allowed. The result must be a valid time (greater than or equal
			 *  to zero milliseconds and less than 24 hours).
			 * </returns>
			 **************************************************************************************************/
			AXISSYSTEMBASE_API Time operator + (const Time& time, const Timespan& timespan);

			/**************************************************************************************************
			 * <summary>	Subtraction operator. </summary>
			 *
			 * <param name="time">	  	The base time. </param>
			 * <param name="timespan">	Amount of time to subtract. </param>
			 *
			 * <returns>
			 *  A new time object earlier than the base time by the amount specified in the timespan
			 *  parameter. Whole days are not allowed. The result must be a valid time (greater than or equal
			 *  to zero milliseconds and less than 24 hours).
			 * </returns>
			 **************************************************************************************************/
			AXISSYSTEMBASE_API Time operator - (const Time& time, const Timespan& timespan);

			/**************************************************************************************************
			 * <summary>	Subtraction operator. </summary>
			 *
			 * <param name="t1">	The base time. </param>
			 * <param name="t2">	The earlier time. </param>
			 *
			 * <returns>
			 *  A Timespan object which stores the time difference between the two Time objects.
			 * </returns>
			 **************************************************************************************************/
			AXISSYSTEMBASE_API Timespan operator - (const Time& t1, const Time& t2);
		}
	}
}

