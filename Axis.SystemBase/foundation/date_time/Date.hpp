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
			 * <summary>	Class for date arithmetics and representation. Objects from this class are
			 * 				immutable although it does allow copy assignment. </summary>
			 **************************************************************************************************/
			class AXISSYSTEMBASE_API Date
			{
			private:
				class DateData;
				DateData *_data;
			public:

				/**************************************************************************************************
				 * <summary>	Values that represent the days of the week. </summary>
				 **************************************************************************************************/
				enum WeekDay
				{
					Sunday    = 0,
					Monday    = 1,
					Tuesday   = 2,
					Wednesday = 3,
					Thursday  = 4,
					Friday    = 5,
					Saturday  = 6
				};

				/**************************************************************************************************
				 * <summary>	Creates a new date object pointing to Jan, 1st 1900. </summary>
				 **************************************************************************************************/
				Date(void);

				/**************************************************************************************************
				 * <summary>	Copy constructor. </summary>
				 *
				 * <param name="other">	The other date object. </param>
				 **************************************************************************************************/
				Date(const axis::foundation::date_time::Date& other);

				/**************************************************************************************************
				 * <summary>	Creates a new date object. </summary>
				 *
				 * <param name="year"> 	The year. </param>
				 * <param name="month">	The month of the year. </param>
				 * <param name="day">  	The day of the month. </param>
				 **************************************************************************************************/
				Date(int year, int month, int day);

				/**************************************************************************************************
				 * <summary>	Destructor. </summary>
				 **************************************************************************************************/
				~Date(void);

				/**************************************************************************************************
				 * <summary>	Returns the day of the month. </summary>
				 *
				 * <returns>	The day of the month. </returns>
				 **************************************************************************************************/
				int GetDay(void) const;

				/**************************************************************************************************
				 * <summary>	Returns the month number. </summary>
				 *
				 * <returns>	The month number. </returns>
				 **************************************************************************************************/
				int GetMonth(void) const;

				/**************************************************************************************************
				 * <summary>	Returns the year. </summary>
				 *
				 * <returns>	The year. </returns>
				 **************************************************************************************************/
				int GetYear(void) const;

				/**************************************************************************************************
				 * <summary>	Returns the day of the week for the current date. </summary>
				 *
				 * <returns>	The day of the week. </returns>
				 **************************************************************************************************/
				WeekDay GetDayOfWeek(void) const;

				/**************************************************************************************************
				 * <summary>	Returns the day of the year for the current date. </summary>
				 *
				 * <returns>	How many days passed since January 1st (inclusive) of the referred year. </returns>
				 **************************************************************************************************/
				int GetDayOfYear(void) const;

				/**************************************************************************************************
				 * <summary>	Returns the ISO 8601 week number for the current date. </summary>
				 *
				 * <returns>	The week index. </returns>
				 **************************************************************************************************/
				int GetWeekOfYear(void) const;

				/**************************************************************************************************
				 * <summary>	Copy assignment operator. </summary>
				 *
				 * <param name="other">	The other date object. </param>
				 *
				 * <returns>	A reference to this object. </returns>
				 **************************************************************************************************/
				Date& operator =(const axis::foundation::date_time::Date& other);

				/**************************************************************************************************
				 * <summary>	Less-than comparison operator. </summary>
				 *
				 * <param name="other">	The other date to compare. </param>
				 *
				 * <returns>	true if this date is earlier than the other one. </returns>
				 **************************************************************************************************/
				bool operator <(const axis::foundation::date_time::Date& other) const;

				/**************************************************************************************************
				 * <summary>	Less-than-or-equal comparison operator. </summary>
				 *
				 * <param name="other">	The other date to compare. </param>
				 *
				 * <returns>	true if this date is earlier than or equal to the other one. </returns>
				 **************************************************************************************************/
				bool operator <=(const axis::foundation::date_time::Date& other) const;

				/**************************************************************************************************
				 * <summary>	Greater-than comparison operator. </summary>
				 *
				 * <param name="other">	The other date to compare. </param>
				 *
				 * <returns>	true if this date is after the other one. </returns>
				 **************************************************************************************************/
				bool operator >(const axis::foundation::date_time::Date& other) const;

				/**************************************************************************************************
				 * <summary>	Greater-than-or-equal comparison operator. </summary>
				 *
				 * <param name="other">	The other date to compare. </param>
				 *
				 * <returns>	true if this date is equal to or after the other one. </returns>
				 **************************************************************************************************/
				bool operator >=(const axis::foundation::date_time::Date& other) const;

				/**************************************************************************************************
				 * <summary>	Equality operator. </summary>
				 *
				 * <param name="other">	The other date to compare. </param>
				 *
				 * <returns>	true if this date is equal to the other one. </returns>
				 **************************************************************************************************/
				bool operator ==(const axis::foundation::date_time::Date& other) const;

				/**************************************************************************************************
				 * <summary>	Inequality operator. </summary>
				 *
				 * <param name="other">	The other date to compare. </param>
				 *
				 * <returns>	true if this date is different from the other one. </returns>
				 **************************************************************************************************/
				bool operator !=(const axis::foundation::date_time::Date& other) const;

				/**************************************************************************************************
				 * <summary>	Returns the current date. </summary>
				 *
				 * <returns>	A date object with the current date. </returns>
				 **************************************************************************************************/
				static Date Today(void);
			};

			/**************************************************************************************************
			 * <summary>	Addition operator. </summary>
			 *
			 * <param name="date">	   	The base date. </param>
			 * <param name="dateRange">	Amount of days to add. </param>
			 *
			 * <returns>
			 *  A new date object farther from the base date by the amount specified in the dateRange
			 *  parameter. Fractions of a day (that is, hours, minutes, seconds and milliseconds) 
			 *  are not allowed.
			 * </returns>
			 **************************************************************************************************/
			AXISSYSTEMBASE_API Date operator + (const axis::foundation::date_time::Date& date, const axis::foundation::date_time::Timespan& dateRange);

			/**************************************************************************************************
			 * <summary>	Subtraction operator. </summary>
			 *
			 * <param name="date">	   	The base date. </param>
			 * <param name="dateRange">	Amount of days to subtract. </param>
			 *
			 * <returns>	
			 *  A new date object earlier from the base date by the amount specified in the dateRange
			 *  parameter. Fractions of a day (that is, hours, minutes, seconds and milliseconds) 
			 *  are not allowed.
			 * </returns>
			 **************************************************************************************************/
			AXISSYSTEMBASE_API Date operator - (const axis::foundation::date_time::Date& date, const axis::foundation::date_time::Timespan& dateRange);

			/**************************************************************************************************
			 * <summary>	Subtraction operator. </summary>
			 *
			 * <param name="d1">	The base date. </param>
			 * <param name="d2">	The earlier date. </param>
			 *
			 * <returns>	The amount of days between the two dates. </returns>
			 **************************************************************************************************/
			AXISSYSTEMBASE_API Timespan operator - (const axis::foundation::date_time::Date& d1, const axis::foundation::date_time::Date& d2);
		}
	}
}

