#pragma once
#include "DateTimeFacet.hpp"
#include <xlocale>

namespace axis
{
	namespace services
	{
		namespace locales
		{
			class SystemDateTimeFacet : public DateTimeFacet
			{
			private:
				std::locale _systemLocale;

				SystemDateTimeFacet(const SystemDateTimeFacet& other);
				SystemDateTimeFacet& operator =(const SystemDateTimeFacet& other);

				virtual axis::String DoToShortDateString( const axis::foundation::date_time::Date& date ) const;

				virtual axis::String DoToLongDateString( const axis::foundation::date_time::Date& date ) const;

				virtual axis::String DoToShortTimeString( const axis::foundation::date_time::Time& time ) const;

				virtual axis::String DoToLongTimeString( const axis::foundation::date_time::Time& time ) const;

				virtual axis::String DoToLongTimeMillisString( const axis::foundation::date_time::Time& time ) const;

				virtual axis::String DoToShortTimeRangeString( const axis::foundation::date_time::Timespan& timespan, axis::foundation::date_time::Timespan::TimespanInterval majorRange ) const;

				virtual axis::String DoToLongTimeRangeString( const axis::foundation::date_time::Timespan& timespan, axis::foundation::date_time::Timespan::TimespanInterval majorRange ) const;

				virtual axis::String DoToShortDateTimeString( const axis::foundation::date_time::Timestamp& timestamp ) const;

				virtual axis::String DoToLongDateTimeString( const axis::foundation::date_time::Timestamp& timestamp ) const;

				virtual axis::String DoToShortDateTimeMillisString( const axis::foundation::date_time::Timestamp& timestamp ) const;

				virtual axis::String DoToLongDateTimeMillisString( const axis::foundation::date_time::Timestamp& timestamp ) const;

				virtual axis::foundation::date_time::Timestamp DoToLocalTime(const axis::foundation::date_time::Timestamp& timestamp) const;
			public:
				SystemDateTimeFacet(void);

			};
		}
	}
}

