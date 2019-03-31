#pragma once
#include "AxisString.hpp"
#include "foundation/Axis.SystemBase.hpp"
#include "foundation/date_time/Date.hpp"
#include "foundation/date_time/Time.hpp"
#include "foundation/date_time/Timespan.hpp"

namespace axis
{
	namespace foundation
	{
		namespace date_time
		{
			/**************************************************************************************************
			 * <summary>
			 *  Stores date and time information about a specific instant of time. Objects of this class are
			 *  immutable although it does allow copy assignments.
			 * </summary>
			 **************************************************************************************************/
			class AXISSYSTEMBASE_API Timestamp
			{
			private:
				Date _date;
				Time _time;
				bool _isLocalTime;
			public:

				/**************************************************************************************************
				 * <summary>	Creates a new timestamp which points to Jan 1st 1900 at midnight in local time. </summary>
				 **************************************************************************************************/
				Timestamp(void);

				/**************************************************************************************************
				 * <summary>
				 *  Creates a new timestamp which points to the midnight of the specified date in local time.
				 * </summary>
				 *
				 * <param name="date">	The date. </param>
				 **************************************************************************************************/
				Timestamp(Date date);

				/**************************************************************************************************
				 * <summary>
				 *  Creates a new timestamp which points to a specific time of Jan 1st 1900 in local time.
				 * </summary>
				 *
				 * <param name="time">	The time. </param>
				 **************************************************************************************************/
				Timestamp(Time time);

				/**************************************************************************************************
				 * <summary>
				 *  Creates a new timestamp pointing to a specific instant of time, represented in local time.
				 * </summary>
				 *
				 * <param name="date">	The date. </param>
				 * <param name="time">	The time. </param>
				 **************************************************************************************************/
				Timestamp(Date date, Time time);

				/**************************************************************************************************
				 * <summary>	Creates a new timestamp pointing to a specific instant of time. </summary>
				 *
				 * <param name="date">	   	The date. </param>
				 * <param name="time">	   	The time. </param>
				 * <param name="isUTCTime">	Tells whether the supplied date/time is a Universal Coordinate Time
				 * 							(UTC). </param>
				 **************************************************************************************************/
				Timestamp(Date date, Time time, bool isUTCTime);

				/**************************************************************************************************
				 * <summary>	Copy constructor. </summary>
				 *
				 * <param name="timestamp">	The other timestamp object. </param>
				 **************************************************************************************************/
				Timestamp(const Timestamp& timestamp);

				/**************************************************************************************************
				 * <summary>	Destructor. </summary>
				 **************************************************************************************************/
				~Timestamp(void);

				/**************************************************************************************************
				 * <summary>	Returns the current local time. </summary>
				 *
				 * <returns>	The local time. </returns>
				 **************************************************************************************************/
				static Timestamp GetLocalTime(void);

				/**************************************************************************************************
				 * <summary>	Returns the current UTC time. </summary>
				 *
				 * <returns>	The UTC time. </returns>
				 **************************************************************************************************/
				static Timestamp GetUTCTime(void);

				/**************************************************************************************************
				 * <summary>	Converts this object to an UTC time. </summary>
				 *
				 * <returns>
				 *  A new timestamp object representing the same instant of time in UTC time. If this object is
				 *  already using UTC time, no conversion is made and a copy of this object is created.
				 * </returns>
				 **************************************************************************************************/
				Timestamp ToUTCTime(void) const;

				/**************************************************************************************************
				 * <summary>	Converts this object to local time. </summary>
				 *
				 * <returns>
				 *  A new timestamp object representing the same instant of time in local time. If this object is
				 *  already using local time, no conversion is made and a copy of this object is created.
				 * </returns>
				 **************************************************************************************************/
				Timestamp ToLocalTime(void) const;

				/**************************************************************************************************
				 * <summary>	Returns the date part of this timestamp. </summary>
				 *
				 * <returns>	The date. </returns>
				 **************************************************************************************************/
				Date GetDate(void) const;

				/**************************************************************************************************
				 * <summary>	Returns the time part of this timestamp. </summary>
				 *
				 * <returns>	The time. </returns>
				 **************************************************************************************************/
				Time GetTime(void) const;

				/**************************************************************************************************
				 * <summary>	Queries if this timestamp is in local time. </summary>
				 *
				 * <returns>	true if it is, false otherwise. </returns>
				 **************************************************************************************************/
				bool IsLocalTime(void) const;

				/**************************************************************************************************
				 * <summary>	Queries if this object is in UTC time. </summary>
				 *
				 * <returns>	true if it is, false otherwise. </returns>
				 **************************************************************************************************/
				bool IsUTCTime(void) const;

				/**************************************************************************************************
				 * <summary>	Copy assignment operator. </summary>
				 *
				 * <param name="other">	The other timestamp object. </param>
				 *
				 * <returns>	A reference to this object. </returns>
				 **************************************************************************************************/
				Timestamp& operator = (const Timestamp& other);
			};

			/**************************************************************************************************
			 * <summary>	Subtraction operator. </summary>
			 *
			 * <param name="t1">	The reference date/time. </param>
			 * <param name="t2">	The date/time to subtract from it. </param>
			 *
			 * <returns>	The amount of time between the two dates. </returns>
			 **************************************************************************************************/
			AXISSYSTEMBASE_API Timespan  operator - (const Timestamp& t1,    const Timestamp& t2);

			/**************************************************************************************************
			 * <summary>	Addition operator. </summary>
			 *
			 * <param name="time"> 	The base date/time. </param>
			 * <param name="range">	A value to add to it. </param>
			 *
			 * <returns>
			 *  A new timestamp object which is farther than the base date/time as specified by the range
			 *  parameter.
			 * </returns>
			 **************************************************************************************************/
			AXISSYSTEMBASE_API Timestamp operator + (const Timestamp& time,  const Timespan& range);

			/**************************************************************************************************
			 * <summary>	Addition operator. </summary>
			 *
			 * <param name="range">	A value to add to it. </param>
			 * <param name="time"> 	The base date/time. </param>
			 *
			 * <returns>
			 *  A new timestamp object which is farther than the base date/time as specified by the range
			 *  parameter.
			 * </returns>
			 **************************************************************************************************/
			AXISSYSTEMBASE_API Timestamp operator + (const Timespan&  range, const Timestamp& time);

			/**************************************************************************************************
			 * <summary>	Subtraction operator. </summary>
			 *
			 * <param name="time"> 	The base date/time. </param>
			 * <param name="range">	A value to subtract from it. </param>
			 *
			 * <returns>
			 *  A new timestamp object which is earlier than the base date/time as specified by the range
			 *  parameter.
			 * </returns>
			 **************************************************************************************************/
			AXISSYSTEMBASE_API Timestamp operator - (const Timestamp& time,  const Timespan& range);
		}
	}
}

