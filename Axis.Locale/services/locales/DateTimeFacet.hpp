#pragma once
#include "foundation/Axis.Locale.hpp"
#include "AxisString.hpp"
#include "foundation/date_time/Date.hpp"
#include "foundation/date_time/Time.hpp"
#include "foundation/date_time/Timespan.hpp"
#include "foundation/date_time/Timestamp.hpp"

namespace axis
{
	namespace services
	{
		namespace locales
		{
			/**
			 * Provides conversion and text generation services for date types
			 * according to a specific locale.
			 */
			class AXISLOCALE_API DateTimeFacet
			{
			private:
				DateTimeFacet(const DateTimeFacet& other);
				DateTimeFacet& operator =(const DateTimeFacet& other);


				/**
				 * Converts a date to a short date string.
				 *
				 * @param date The date.
				 *
				 * @return The short date string representation according to the locale.
				 */
				virtual axis::String DoToShortDateString(const axis::foundation::date_time::Date& date) const;

				/**
				 * Converts a date to a long date string.
				 *
				 * @param date The date.
				 *
				 * @return The long date string representation according to the locale.
				 */
				virtual axis::String DoToLongDateString(const axis::foundation::date_time::Date& date) const;

				/**
				 * Converts a time to a short time string.
				 *
				 * @param time The time.
				 *
				 * @return The short time string representation according to the locale.
				 */
				virtual axis::String DoToShortTimeString(const axis::foundation::date_time::Time& time) const;

				/**
				 * Converts a time to a long time string.
				 *
				 * @param time The time.
				 *
				 * @return The long time string representation according to the locale.
				 */
				virtual axis::String DoToLongTimeString(const axis::foundation::date_time::Time& time) const;

				/**
				 * Converts a time to a long time string considering fractional seconds.
				 *
				 * @param time The time.
				 *
				 * @return The long time string (with milliseconds) representation according to the locale.
				 */
				virtual axis::String DoToLongTimeMillisString(const axis::foundation::date_time::Time& time) const;

				/**
				 * Converts a timespan to a short time range string.
				 *
				 * @param timespan   The timespan.
				 * @param majorRange (optional) tells the greatest range granularity that 
				 * 					 should be considered when converting.
				 *
				 * @return The short time range string representation according to the locale.
				 */
				virtual axis::String DoToShortTimeRangeString(const axis::foundation::date_time::Timespan& timespan, axis::foundation::date_time::Timespan::TimespanInterval majorRange) const;

				/**
				 * Converts a timespan to a long time range string.
				 *
				 * @param timespan   The timespan.
				 * @param majorRange (optional) tells the greatest range granularity that 
				 * 					 should be considered when converting.
				 *
				 * @return The long time range string representation according to the locale.
				 */
				virtual axis::String DoToLongTimeRangeString(const axis::foundation::date_time::Timespan& timespan, axis::foundation::date_time::Timespan::TimespanInterval majorRange) const;

				/**
				 * Converts a timestamp to a short date/time string.
				 *
				 * @param timestamp The timestamp.
				 *
				 * @return The short date/time string representation according to the locale.
				 */
				virtual axis::String DoToShortDateTimeString(const axis::foundation::date_time::Timestamp& timestamp) const;

				/**
				 * Converts a timestamp to a long date/time string.
				 *
				 * @param timestamp The timestamp.
				 *
				 * @return The long date/time string representation according to the locale.
				 */
				virtual axis::String DoToLongDateTimeString(const axis::foundation::date_time::Timestamp& timestamp) const;

				/**
				 * Converts a timestamp to a short date/time string considering fractional seconds.
				 *
				 * @param timestamp The timestamp.
				 *
				 * @return The short date/time string representation (with milliseconds) 
				 * 		   according to the locale.
				 */
				virtual axis::String DoToShortDateTimeMillisString(const axis::foundation::date_time::Timestamp& timestamp) const;

				/**
				 * Converts a timestamp to a long date/time string considering fractional seconds.
				 *
				 * @param timestamp The timestamp.
				 *
				 * @return The long date/time string representation (with milliseconds) 
				 * 		   according to the locale.
				 */
				virtual axis::String DoToLongDateTimeMillisString(const axis::foundation::date_time::Timestamp& timestamp) const;

				/**
				 * Converts a timestamp in UTC to local time.
				 *
				 * @param timestamp The timestamp.
				 *
				 * @return A new timestamp object representing the same instant of time using
				 * 		   timezone settings for the referred locale.
				 */
				virtual axis::foundation::date_time::Timestamp DoToLocalTime(const axis::foundation::date_time::Timestamp& timestamp) const;
			public:

				/**
				 * Default constructor.
				 */
				DateTimeFacet(void);

				/**
				 * Destructor.
				 */
				virtual ~DateTimeFacet(void);

				/**
				 * Converts a date to a short date string.
				 *
				 * @param date The date.
				 *
				 * @return The short date string representation according to the locale.
				 */
				axis::String ToShortDateString(const axis::foundation::date_time::Date& date) const;

				/**
				 * Converts a date to a long date string.
				 *
				 * @param date The date.
				 *
				 * @return The long date string representation according to the locale.
				 */
				axis::String ToLongDateString(const axis::foundation::date_time::Date& date) const;

				/**
				 * Converts a time to a short time string.
				 *
				 * @param time The time.
				 *
				 * @return The short time string representation according to the locale.
				 */
				axis::String ToShortTimeString(const axis::foundation::date_time::Time& time) const;

				/**
				 * Converts a time to a long time string.
				 *
				 * @param time The time.
				 *
				 * @return The long time string representation according to the locale.
				 */
				axis::String ToLongTimeString(const axis::foundation::date_time::Time& time) const;

				/**
				 * Converts a time to a long time string considering fractional seconds.
				 *
				 * @param time The time.
				 *
				 * @return The long time string (with milliseconds) representation according to the locale.
				 */
				axis::String ToLongTimeMillisString(const axis::foundation::date_time::Time& time) const;

				/**
				 * Converts a timespan to a short time range string.
				 *
				 * @param timespan   The timespan.
				 * @param majorRange (optional) tells the greatest range granularity that 
				 * 					 should be considered when converting.
				 *
				 * @return The short time range string representation according to the locale.
				 */
				axis::String ToShortTimeRangeString(const axis::foundation::date_time::Timespan& timespan, axis::foundation::date_time::Timespan::TimespanInterval majorRange = axis::foundation::date_time::Timespan::Days) const;

				/**
				 * Converts a timespan to a long time range string.
				 *
				 * @param timespan   The timespan.
				 * @param majorRange (optional) tells the greatest range granularity that 
				 * 					 should be considered when converting.
				 *
				 * @return The long time range string representation according to the locale.
				 */
				axis::String ToLongTimeRangeString(const axis::foundation::date_time::Timespan& timespan, axis::foundation::date_time::Timespan::TimespanInterval majorRange = axis::foundation::date_time::Timespan::Days) const;

				/**
				 * Converts a timestamp to a short date/time string.
				 *
				 * @param timestamp The timestamp.
				 *
				 * @return The short date/time string representation according to the locale.
				 */
				axis::String ToShortDateTimeString(const axis::foundation::date_time::Timestamp& timestamp) const;

				/**
				 * Converts a timestamp to a long date/time string.
				 *
				 * @param timestamp The timestamp.
				 *
				 * @return The long date/time string representation according to the locale.
				 */
				axis::String ToLongDateTimeString(const axis::foundation::date_time::Timestamp& timestamp) const;

				/**
				 * Converts a timestamp to a short date/time string considering fractional seconds.
				 *
				 * @param timestamp The timestamp.
				 *
				 * @return The short date/time string representation (with milliseconds) 
				 * 		   according to the locale.
				 */
				axis::String ToShortDateTimeMillisString(const axis::foundation::date_time::Timestamp& timestamp) const;

				/**
				 * Converts a timestamp to a long date/time string considering fractional seconds.
				 *
				 * @param timestamp The timestamp.
				 *
				 * @return The long date/time string representation (with milliseconds) 
				 * 		   according to the locale.
				 */
				axis::String ToLongDateTimeMillisString(const axis::foundation::date_time::Timestamp& timestamp) const;

				/**
				 * Converts a timestamp in UTC to local time.
				 *
				 * @param timestamp The timestamp.
				 *
				 * @return A new timestamp object representing the same instant of time using
				 * 		   timezone settings for the referred locale.
				 */
				axis::foundation::date_time::Timestamp ToLocalTime(const axis::foundation::date_time::Timestamp& timestamp) const;
			};
		}
	}
}

