#pragma once
#include "foundation/Axis.SystemBase.hpp"

namespace axis
{
	namespace foundation
	{
		namespace date_time
		{
			/**************************************************************************************************
			 * <summary>	Represents a time lapse between two reference points in time. </summary>
			 **************************************************************************************************/
			class AXISSYSTEMBASE_API Timespan
			{
			public:

				/**************************************************************************************************
				 * <summary>	Values that represent possible timespan interval components. </summary>
				 **************************************************************************************************/
				enum TimespanInterval
				{
					Days,
					Hours,
					Minutes,
					Seconds,
					Milliseconds
				};

				class TimespanData;
				TimespanData *_data;

				/**************************************************************************************************
				 * <summary>	Creates a new timespan object with null time lapse (zero milliseconds). </summary>
				 **************************************************************************************************/
				Timespan(void);

				/**************************************************************************************************
				 * <summary>	Copy constructor. </summary>
				 *
				 * <param name="timespan">	The other timespan object. </param>
				 **************************************************************************************************/
				Timespan(const Timespan& timespan);

				/**************************************************************************************************
				 * <summary>	Creates a new timespan object. </summary>
				 *
				 * <param name="hours">  	Amount of whole hours to add to the range. </param>
				 * <param name="minutes">	Amount of whole minutes to add to the range. </param>
				 * <param name="seconds">	Amount of whole seconds to add to the range. </param>
				 **************************************************************************************************/
				Timespan(long hours, long minutes, long seconds);

				/**************************************************************************************************
				 * <summary>	Creates a new timespan object. </summary>
				 *
				 * <param name="hours">		  	Amount of whole hours to add to the range. </param>
				 * <param name="minutes">	  	Amount of whole minutes to add to the range. </param>
				 * <param name="seconds">	  	Amount of whole seconds to add to the range. </param>
				 * <param name="milliseconds">	Amount of whole milliseconds to add to the range. </param>
				 **************************************************************************************************/
				Timespan(long hours, long minutes, long seconds, long milliseconds);

				/**************************************************************************************************
				 * <summary>	Creates a new timespan object. </summary>
				 *
				 * <param name="days">		  	Amount of whole days to add to the range. </param>
				 * <param name="hours">		  	Amount of whole hours to add to the range. </param>
				 * <param name="minutes">	  	Amount of whole minutes to add to the range. </param>
				 * <param name="seconds">	  	Amount of whole seconds to add to the range. </param>
				 * <param name="milliseconds">	Amount of whole milliseconds to add to the range. </param>
				 **************************************************************************************************/
				Timespan(long days, long hours, long minutes, long seconds, long milliseconds);

				/**************************************************************************************************
				 * <summary>	Creates a new timespan object. </summary>
				 *
				 * <param name="value">		  	Amount to be considered. </param>
				 * <param name="intervaltype">	Tells how to interpret the amount specified in the previous parameter. </param>
				 **************************************************************************************************/
				Timespan(long value, TimespanInterval intervaltype);

				/**************************************************************************************************
				 * <summary>	Creates a new timespan object. </summary>
				 *
				 * <param name="tenthsOfMilliseconds">	The interval represented in tenths of milliseconds. </param>
				 **************************************************************************************************/
				Timespan(uint64 tenthsOfMilliseconds);

				/**************************************************************************************************
				 * <summary>	Destructor. </summary>
				 **************************************************************************************************/
				~Timespan(void);

				/**************************************************************************************************
				 * <summary>	Returns the whole days in the range. </summary>
				 *
				 * <returns>	The days. </returns>
				 **************************************************************************************************/
				long GetDays(void) const;

				/**************************************************************************************************
				 * <summary>	Returns the hours of a day in the range. </summary>
				 *
				 * <returns>	The hours. </returns>
				 **************************************************************************************************/
				long GetHours(void) const;

				/**************************************************************************************************
				 * <summary>	Returns the minutes of an hour in the range. </summary>
				 *
				 * <returns>	The minutes. </returns>
				 **************************************************************************************************/
				long GetMinutes(void) const;

				/**************************************************************************************************
				 * <summary>	Returns the seconds of a minute in the range. </summary>
				 *
				 * <returns>	The seconds. </returns>
				 **************************************************************************************************/
				long GetSeconds(void) const;

				/**************************************************************************************************
				 * <summary>	Returns the milliseconds of a second in the range. </summary>
				 *
				 * <returns>	The milliseconds. </returns>
				 **************************************************************************************************/
				long GetMilliseconds(void) const;

				/**************************************************************************************************
				 * <summary>	Returns the total whole hours in the range. </summary>
				 *
				 * <returns>	The total hours. </returns>
				 **************************************************************************************************/
				long GetTotalHours(void) const;

				/**************************************************************************************************
				 * <summary>	Returns the total whole minutes in the range. </summary>
				 *
				 * <returns>	The total minutes. </returns>
				 **************************************************************************************************/
				long GetTotalMinutes(void) const;

				/**************************************************************************************************
				 * <summary>	Returns the total whole seconds in the range. </summary>
				 *
				 * <returns>	The total seconds. </returns>
				 **************************************************************************************************/
				long GetTotalSeconds(void) const;

				/**************************************************************************************************
				 * <summary>	Returns the total milliseconds in the range. </summary>
				 *
				 * <returns>	The total milliseconds. </returns>
				 **************************************************************************************************/
				long GetTotalMilliseconds(void) const;

				/**************************************************************************************************
				 * <summary>	Copy assignment operator. </summary>
				 *
				 * <param name="t">	The other timespan object. </param>
				 *
				 * <returns>	A reference to this object. </returns>
				 **************************************************************************************************/
				Timespan& operator = (const Timespan& t);

				/**************************************************************************************************
				 * <summary>	Less-than comparison operator. </summary>
				 *
				 * <param name="other">	The other timespan to compare. </param>
				 *
				 * <returns>	true if this timespan is less than the other one. </returns>
				 **************************************************************************************************/
				bool operator < (const Timespan& other) const;

				/**************************************************************************************************
				 * <summary>	Less-than-or-equal comparison operator. </summary>
				 *
				 * <param name="other">	The other timespan to compare. </param>
				 *
				 * <returns>	true if this timespan is less than or equal to the other one. </returns>
				 **************************************************************************************************/
				bool operator <=(const Timespan& other) const;

				/**************************************************************************************************
				 * <summary>	Greater-than comparison operator. </summary>
				 *
				 * <param name="other">	The other timespan to compare. </param>
				 *
				 * <returns>	true if this timespan is greater than the other one. </returns>
				 **************************************************************************************************/
				bool operator > (const Timespan& other) const;

				/**************************************************************************************************
				 * <summary>	Greater-than-or-equal comparison operator. </summary>
				 *
				 * <param name="other">	The other timespan to compare. </param>
				 *
				 * <returns>	true if this timespan is greater than or equal to the other one. </returns>
				 **************************************************************************************************/
				bool operator >=(const Timespan& other) const;

				/**************************************************************************************************
				 * <summary>	Equality operator. </summary>
				 *
				 * <param name="other">	The other timespan to compare. </param>
				 *
				 * <returns>	true if this timespan is equal to the other one. </returns>
				 **************************************************************************************************/
				bool operator ==(const Timespan& other) const;

				/**************************************************************************************************
				 * <summary>	Inequality operator. </summary>
				 *
				 * <param name="other">	The other timespan to compare. </param>
				 *
				 * <returns>	true if this timespan is different from the other one. </returns>
				 **************************************************************************************************/
				bool operator !=(const Timespan& other) const;

				/**************************************************************************************************
				 * <summary>	Queries if this object has fractions of a day. </summary>
				 *
				 * <returns>	true if it has, false otherwise. </returns>
				 **************************************************************************************************/
				bool HasFractionalDay(void) const;

				/**************************************************************************************************
				 * <summary>	Queries if this object has whole days. </summary>
				 *
				 * <returns>	true if it has, false otherwise. </returns>
				 **************************************************************************************************/
				bool HasWholeDays(void) const;

				/**************************************************************************************************
				 * <summary>	Returns the whole days of this object. </summary>
				 *
				 * <returns>	A new timespan object containing only the whole days. </returns>
				 **************************************************************************************************/
				Timespan GetWholeDays(void) const;

				/**************************************************************************************************
				 * <summary>	Returns the fractions of a day of this object. </summary>
				 *
				 * <returns>	A new timespan object containing only the fractions of a day. </returns>
				 **************************************************************************************************/
				Timespan GetFractionalDay(void) const;
			};

			/**************************************************************************************************
			 * <summary>	Addition operator. </summary>
			 *
			 * <param name="t1">	The first value. </param>
			 * <param name="t2">	A value to add to it. </param>
			 *
			 * <returns>	A new timespan object which is the sum of the intervals. </returns>
			 **************************************************************************************************/
			AXISSYSTEMBASE_API Timespan operator + (const Timespan& t1, const Timespan& t2);

			/**************************************************************************************************
			 * <summary>	Subtraction operator. </summary>
			 *
			 * <param name="t1">	The first value. </param>
			 * <param name="t2">	A value to subtract from it. </param>
			 *
			 * <returns>	A new timespan object which is the difference of the intervals. </returns>
			 **************************************************************************************************/
			AXISSYSTEMBASE_API Timespan operator - (const Timespan& t1, const Timespan& t2);

			/**************************************************************************************************
			 * <summary>	Multiplication operator. </summary>
			 *
			 * <param name="t1">		The timespan interval. </param>
			 * <param name="factor">	The factor by which the timespan will be multiplied. </param>
			 *
			 * <returns>
			 *  A new timespan object which is equal to the timespan interval multiplied by the specified
			 *  factor.
			 * </returns>
			 **************************************************************************************************/
			AXISSYSTEMBASE_API Timespan operator * (const Timespan& t1, int factor);

			/**************************************************************************************************
			 * <summary>	Multiplication operator. </summary>
			 *
			 * <param name="t1">		The timespan interval. </param>
			 * <param name="factor">	The factor by which the timespan will be multiplied. </param>
			 *
			 * <returns>
			 *  A new timespan object which is equal to the timespan interval multiplied by the specified
			 *  factor.
			 * </returns>
			 **************************************************************************************************/
			AXISSYSTEMBASE_API Timespan operator * (const Timespan& t1, real factor);

			/**************************************************************************************************
			 * <summary>	Multiplication operator. </summary>
			 *
			 * <param name="factor">	The factor by which the timespan will be multiplied. </param>
			 * <param name="t1">		The timespan interval. </param>
			 *
			 * <returns>
			 *  A new timespan object which is equal to the timespan interval multiplied by the specified
			 *  factor.
			 * </returns>
			 **************************************************************************************************/
			AXISSYSTEMBASE_API Timespan operator * (int factor, const Timespan& t1);

			/**************************************************************************************************
			 * <summary>	Multiplication operator. </summary>
			 *
			 * <param name="factor">	The factor by which the timespan will be multiplied. </param>
			 * <param name="t1">		The timespan interval. </param>
			 *
			 * <returns>
			 *  A new timespan object which is equal to the timespan interval multiplied by the specified
			 *  factor.
			 * </returns>
			 **************************************************************************************************/
			AXISSYSTEMBASE_API Timespan operator * (real factor, const Timespan& t1);

			/**************************************************************************************************
			 * <summary>	Division operator. </summary>
			 *
			 * <param name="t1">		The timespan interval. </param>
			 * <param name="factor">	The amount by which the timespan will be divided. </param>
			 *
			 * <returns>
			 *  A new timespan object which is equal to the timespan interval divided the specified factor.
			 * </returns>
			 **************************************************************************************************/
			AXISSYSTEMBASE_API Timespan operator / (const Timespan& t1, int factor);

			/**************************************************************************************************
			 * <summary>	Division operator. </summary>
			 *
			 * <param name="t1">		The timespan interval. </param>
			 * <param name="factor">	The amount by which the timespan will be divided. </param>
			 *
			 * <returns>
			 *  A new timespan object which is equal to the timespan interval divided the specified factor.
			 * </returns>
			 **************************************************************************************************/
			AXISSYSTEMBASE_API Timespan operator / (const Timespan& t1, real factor);
		}
	}
}

